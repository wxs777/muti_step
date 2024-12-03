import copy
import numpy as np

from algorithm.eda_model.edamodel import VWH, local_search
from algorithm.eda_model.utils_eda import float_lhs_init, gen_float_solution_via_matrix, get_data_from_population


# from pyemo.util.population_init import float_lhs_init
# from pyemo.problem.singleobjproblem.lzg import LZG03 as Ackley
# from pyemo.problem.singleobjproblem.lzg import LZG01

class EDALS(object):
    def __init__(self, v_num=None, lb=None, ub=None, Pb=0.2, Pc=0.2, M=10, population_size=50, max_fes=8000):
        self.population_size = population_size
        if v_num is None:
            raise ValueError('v_num is None')
        self.v_num = v_num

        if lb is None:
            raise ValueError('lb is None')

        self.lb = lb

        if ub is None:
            raise ValueError('ub is None')

        self.ub = ub

        self.Pc = Pc
        self.Pb = Pb
        self.M = M
        self.NL = int(np.floor(self.population_size * self.Pb))

        self.EDA = VWH(
            M=self.M,
            D=self.v_num,
            LB=np.array(self.lb),
            UB=np.array(self.ub)
        )

        self.pop_init = False
        self.alpha = self.population_size  # 初始化种群大小

        self.to_eva_list = []
        self.eva_num = 0
        self.max_fes = max_fes

    def is_stop(self):
        return self.eva_num > self.max_fes

    def population_init(self):
        population = float_lhs_init(self.lb, self.ub, self.v_num, 0, self.alpha)
        return population

    def ask(self):
        if not self.pop_init:
            self.to_eva_list = self.population_init()
            return [p.decs for p in self.to_eva_list]
        else:
            current_pop = self.current_pop
            # 产生新解
            new_decs = self.reproduction(current_pop)
            return [x for x in new_decs.tolist()]

    def save_react_chain(self, content, save_path):
        with open(save_path, "a+") as f:
            f.write(content + "\n")

    def tell(self, Xs, ys):
        new_population = gen_float_solution_via_matrix(self.lb, self.ub, objectives_num=self.v_num, constraints_num=0,
                                                       Decs=Xs, Objs=ys)
        self.eva_num += len(new_population)

        if not self.pop_init:
            self.pop_init = True
            self.current_pop = new_population
        else:
            # self.save_react_chain('{}'.format(self.get_current_pop(self.current_pop)), 'ga/react_chain/demo_d_current_pop-03-02.txt')
            # self.save_react_chain('{}'.format(self.get_current_pop(new_population)), 'ga/react_chain/demo_d_new_population-03-02.txt')
            self.current_pop = self.selection(self.current_pop, new_population)
        return self.current_pop

    def get_current_pop(self, population):
        cur_p = copy.deepcopy(population)
        cur_pop = []
        for i, content in enumerate(cur_p):
            cur_pop.append([cur_p[i].decs, cur_p[i].objs])
        return cur_pop

    def get_best(self):
        t_p = copy.deepcopy(self.current_pop)
        t_p.sort(key=lambda s: s.objs[0] * s.objs[1])
        print('t_p:', t_p)
        return t_p[0].decs, t_p[0].objs

    def get_mean(self):
        t_p = copy.deepcopy(self.current_pop)
        all_fx = []
        for line in t_p:
            all_fx.append(line.objs)
        return np.mean(all_fx)

    def selection(self, f_pop, son_pop):
        """
        Select individuals based on non-dominated sorting and crowding distance.
        """
        f_pop.extend(son_pop)
        fronts = self.non_dominated_sort(f_pop)
        selected_population = []

        for front in fronts:
            if len(selected_population) + len(front) <= self.population_size:
                selected_population.extend(front)
            else:
                # 如果加上当前front会超过种群大小，根据拥挤度选择
                # 获取每个个体与其拥挤度
                crowded_front = self.crowding_distance(front)
                # 按拥挤度从大到小排序，保留最优的个体
                sorted_front = [x for _, x in sorted(crowded_front, key=lambda x: x[0], reverse=True)]
                remaining_slots = self.population_size - len(selected_population)
                selected_population.extend(sorted_front[:remaining_slots])
                break

        return selected_population[:self.population_size]

    def selection_de(self, f_pop, son_pop):
        select_result = []
        for i in range(len(f_pop)):
            if f_pop[i].objs >= son_pop[i].objs:
                select_result.append(f_pop[i])
            else:
                select_result.append(son_pop[i])
        return select_result

    def get_name(self):
        return 'EDALS'

    def reproduction(self, pop):
        Xs = get_data_from_population(pop, 'decs')
        ys = get_data_from_population(pop, 'objs')

        I = np.argsort(ys[:, 0])
        Xs = Xs[I]
        ys = ys[I]
        self.EDA.update(Xs, ys)

        Xs_new = self.EDA.sample(self.population_size)

        Xs_l = local_search(Xs[:self.NL, :], ys[:self.NL, 0])
        I = np.floor(np.random.random((self.population_size, 1)) * (Xs_l.shape[0] - 2)).astype(int).flatten()
        xtmp = Xs_l[I, :]
        mask = np.random.random((self.population_size, self.v_num)) < self.Pc
        Xs_new[mask] = xtmp[mask]

        # boundary checking
        lb_matrix = self.lb * np.ones(shape=Xs_new.shape)
        ub_matrix = self.ub * np.ones(shape=Xs_new.shape)
        pos = Xs_new < self.lb
        Xs_new[pos] = 0.5 * (Xs[pos] + lb_matrix[pos])
        pos = Xs_new > self.ub
        Xs_new[pos] = 0.5 * (Xs[pos] + ub_matrix[pos])

        return Xs_new

    # if __name__ == '__main__':
    # # problem
    # prob = LZG01(10)
    # v_num = prob.variables_num
    # lb = prob.lb
    # ub = prob.ub
    #
    # # lb = [-1e10 for _ in range(prob.variables_num)]
    # # ub = [1e10 for _ in range(prob.variables_num)]
    #
    # # algorithm
    # opt = EDALS(v_num=v_num, lb=lb, ub=ub, max_fes=8000)
    #
    # # opt
    # iter_num = 0
    # while not opt.is_stop():
    #     solutions = opt.ask()
    #     observe = [prob.obj_func(x) for x in solutions]
    #     opt.tell(solutions, observe)
    #     Xs, ys = opt.get_best()
    #     print('after {} itre the best value is {}'.format(iter_num, ys))
    #     iter_num += 1
    def non_dominated_sort(self, population):
        """
        Perform non-dominated sorting.
        Returns a list of fronts (each front is a list of individuals).
        """
        fronts = [[]]  # First front (non-dominated individuals)
        for i in range(len(population)):
            population[i].domination_count = 0
            population[i].dominated_solutions = []

            for j in range(len(population)):
                if self.dominates(population[i], population[j]):
                    population[i].dominated_solutions.append(population[j])
                elif self.dominates(population[j], population[i]):
                    population[i].domination_count += 1

            if population[i].domination_count == 0:
                fronts[0].append(population[i])

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for ind in fronts[i]:
                for dominated_ind in ind.dominated_solutions:
                    dominated_ind.domination_count -= 1
                    if dominated_ind.domination_count == 0:
                        next_front.append(dominated_ind)
            i += 1
            fronts.append(next_front)
        return fronts

    def dominates(self, individual1, individual2):
        """
        Return True if individual1 dominates individual2, else False.
        """
        return (individual1.objs[0] <= individual2.objs[0] and individual1.objs[1] <= individual2.objs[1] and
                (individual1.objs[0] < individual2.objs[0] or individual1.objs[1] < individual2.objs[1]))

    def crowding_distance(self, front):
        """
        Calculate the crowding distance for each individual in a front.
        """
        distance = [0] * len(front)

        if len(front) == 0:
            return distance

        # Sort by first objective
        front.sort(key=lambda s: s.objs[0])
        distance[0] = distance[-1] = float('inf')  # 赋予边界的个体无限拥挤度
        for i in range(1, len(front) - 1):
            distance[i] += (front[i + 1].objs[0] - front[i - 1].objs[0])

        # Sort by second objective
        front.sort(key=lambda s: s.objs[1])
        for i in range(1, len(front) - 1):
            distance[i] += (front[i + 1].objs[1] - front[i - 1].objs[1])

        # Return (distance, individual) pairs for sorting
        return [(d, ind) for d, ind in zip(distance, front)]
