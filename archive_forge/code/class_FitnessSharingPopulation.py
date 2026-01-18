from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
class FitnessSharingPopulation(Population):
    """ Fitness sharing population that penalizes structures if they are
    too similar. This is determined by a distance measure

    Parameters:

    comp_key: string
        Key where the distance measure can be found in the
        atoms.info['key_value_pairs'] dictionary.

    threshold: float or int
        Value above which no penalization of the fitness takes place

    alpha_sh: float or int
        Determines the shape of the sharing function.
        Default is 1, which gives a linear sharing function.

    """

    def __init__(self, data_connection, population_size, comp_key, threshold, alpha_sh=1.0, comparator=None, logfile=None, use_extinct=False):
        self.comp_key = comp_key
        self.dt = threshold
        self.alpha_sh = alpha_sh
        self.fit_scaling = 1.0
        self.sh_cache = dict()
        Population.__init__(self, data_connection, population_size, comparator, logfile, use_extinct)

    def __get_fitness__(self, candidates):
        """Input should be sorted according to raw_score."""
        max_s = get_raw_score(candidates[0])
        min_s = get_raw_score(candidates[-1])
        T = min_s - max_s
        shared_fit = []
        for c in candidates:
            sc = get_raw_score(c)
            obj_fit = 0.5 * (1.0 - tanh(2.0 * (sc - max_s) / T - 1.0))
            m = 1.0
            ck = c.info['key_value_pairs'][self.comp_key]
            for other in candidates:
                if other != c:
                    name = tuple(sorted([c.info['confid'], other.info['confid']]))
                    if name not in self.sh_cache:
                        ok = other.info['key_value_pairs'][self.comp_key]
                        d = abs(ck - ok)
                        if d < self.dt:
                            v = 1 - (d / self.dt) ** self.alpha_sh
                            self.sh_cache[name] = v
                        else:
                            self.sh_cache[name] = 0
                    m += self.sh_cache[name]
            shf = obj_fit ** self.fit_scaling / m
            shared_fit.append(shf)
        return shared_fit

    def update(self):
        """ The update method in Population will add to the end of
        the population, that can't be used here since the shared fitness
        will change for all candidates when new are added, therefore
        just recalc the population every time. """
        self.pop = []
        self.__initialize_pop__()
        self._write_log()

    def __initialize_pop__(self):
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)
        if len(all_cand) > 0:
            shared_fit = self.__get_fitness__(all_cand)
            all_sorted = list(zip(*sorted(zip(shared_fit, all_cand), reverse=True)))[1]
            i = 0
            while i < len(all_sorted) and len(self.pop) < self.pop_size:
                c = all_sorted[i]
                i += 1
                eq = False
                for a in self.pop:
                    if self.comparator.looks_like(a, c):
                        eq = True
                        break
                if not eq:
                    self.pop.append(c)
            for a in self.pop:
                a.info['looks_like'] = count_looks_like(a, all_cand, self.comparator)
        self.all_cand = all_cand

    def get_two_candidates(self):
        """ Returns two candidates for pairing employing the
            fitness criteria from
            L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
            and the roulete wheel selection scheme described in
            R.L. Johnston Dalton Transactions,
            Vol. 22, No. 22. (2003), pp. 4193-4207
        """
        if len(self.pop) < 2:
            self.update()
        if len(self.pop) < 2:
            return None
        fit = self.__get_fitness__(self.pop)
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