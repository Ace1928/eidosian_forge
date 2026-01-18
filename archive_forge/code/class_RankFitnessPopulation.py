from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
class RankFitnessPopulation(Population):
    """ Ranks the fitness relative to set variable to flatten the surface
        in a certain direction such that mating across variable is equally
        likely irrespective of raw_score.

        Parameters:

        variable_function: function
            A function that takes as input an Atoms object and returns
            the variable that differentiates the ranks.

        exp_function: boolean
            If True use an exponential function for ranking the fitness.
            If False use the same as in Population. Default True.

        exp_prefactor: float
            The prefactor used in the exponential fitness scaling function.
            Default 0.5
    """

    def __init__(self, data_connection, population_size, variable_function, comparator=None, logfile=None, use_extinct=False, exp_function=True, exp_prefactor=0.5):
        self.exp_function = exp_function
        self.exp_prefactor = exp_prefactor
        self.vf = variable_function
        self.current_fitness = None
        Population.__init__(self, data_connection, population_size, comparator, logfile, use_extinct)

    def get_rank(self, rcand, key=None):
        ordered = list(zip(range(len(rcand)), rcand))
        rec_nic = []
        rank_fit = []
        for o, c in ordered:
            if o not in rec_nic:
                ntr = []
                ce1 = self.vf(c)
                rec_nic.append(o)
                ntr.append([o, c])
                for oother, cother in ordered:
                    if oother not in rec_nic:
                        ce2 = self.vf(cother)
                        if ce1 == ce2:
                            rec_nic.append(oother)
                            ntr.append([oother, cother])
                ntr.sort(key=lambda x: x[1].info['key_value_pairs'][key], reverse=True)
                start_rank = -1
                cor = 0
                for on, cn in ntr:
                    rank = start_rank - cor
                    rank_fit.append([on, cn, rank])
                    cor += 1
        rank_fit.sort(key=itemgetter(0), reverse=False)
        return np.array(list(zip(*rank_fit))[2])

    def __get_fitness__(self, candidates):
        expf = self.exp_function
        rfit = self.get_rank(candidates, key='raw_score')
        if not expf:
            rmax = max(rfit)
            rmin = min(rfit)
            T = rmin - rmax
            msg = 'Equal fitness for best and worst candidate in the '
            msg += 'population! Fitness scaling is impossible! '
            msg += 'Try with a larger population.'
            assert T != 0.0, msg
            return 0.5 * (1.0 - np.tanh(2.0 * (rfit - rmax) / T - 1.0))
        else:
            return self.exp_prefactor ** (-rfit - 1)

    def update(self):
        """ The update method in Population will add to the end of
        the population, that can't be used here since the fitness
        will potentially change for all candidates when new are added,
        therefore just recalc the population every time. """
        self.pop = []
        self.__initialize_pop__()
        self.current_fitness = self.__get_fitness__(self.pop)
        self._write_log()

    def __initialize_pop__(self):
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)
        if len(all_cand) > 0:
            fitf = self.__get_fitness__(all_cand)
            all_sorted = list(zip(fitf, all_cand))
            all_sorted.sort(key=itemgetter(0), reverse=True)
            sort_cand = []
            for _, t2 in all_sorted:
                sort_cand.append(t2)
            all_sorted = sort_cand
            i = 0
            while i < len(all_sorted) and len(self.pop) < self.pop_size:
                c = all_sorted[i]
                c_vf = self.vf(c)
                i += 1
                eq = False
                for a in self.pop:
                    a_vf = self.vf(a)
                    if a_vf == c_vf:
                        if self.comparator.looks_like(a, c):
                            eq = True
                            break
                if not eq:
                    self.pop.append(c)
        self.all_cand = all_cand

    def get_two_candidates(self):
        """ Returns two candidates for pairing employing the
            roulete wheel selection scheme described in
            R.L. Johnston Dalton Transactions,
            Vol. 22, No. 22. (2003), pp. 4193-4207
        """
        if len(self.pop) < 2:
            self.update()
        if len(self.pop) < 2:
            return None
        fit = self.current_fitness
        fmax = max(fit)
        c1 = self.pop[0]
        c2 = self.pop[0]
        while c1.info['confid'] == c2.info['confid']:
            nnf = True
            while nnf:
                t = self.rng.randint(len(self.pop))
                if fit[t] > self.rng.rand() * fmax:
                    c1 = self.pop[t]
                    nnf = False
            nnf = True
            while nnf:
                t = self.rng.randint(len(self.pop))
                if fit[t] > self.rng.rand() * fmax:
                    c2 = self.pop[t]
                    nnf = False
        return (c1.copy(), c2.copy())