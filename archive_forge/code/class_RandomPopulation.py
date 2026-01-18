from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
class RandomPopulation(Population):

    def __init__(self, data_connection, population_size, comparator=None, logfile=None, exclude_used_pairs=False, bad_candidates=0, use_extinct=False):
        self.exclude_used_pairs = exclude_used_pairs
        self.bad_candidates = bad_candidates
        Population.__init__(self, data_connection, population_size, comparator, logfile, use_extinct)

    def __initialize_pop__(self):
        """ Private method that initializes the population when
            the population is created. """
        ue = self.use_extinct
        all_cand = self.dc.get_all_relaxed_candidates(use_extinct=ue)
        all_cand.sort(key=lambda x: get_raw_score(x), reverse=True)
        if len(all_cand) > 0:
            ratings = []
            best_raw = get_raw_score(all_cand[0])
            i = 0
            while i < len(all_cand):
                c = all_cand[i]
                i += 1
                eq = False
                for a in self.pop:
                    if self.comparator.looks_like(a, c):
                        eq = True
                        break
                if not eq:
                    if len(self.pop) < self.pop_size - self.bad_candidates:
                        self.pop.append(c)
                    else:
                        exp_fact = exp(get_raw_score(c) / best_raw)
                        ratings.append([c, (exp_fact - 1) * self.rng.rand()])
            ratings.sort(key=itemgetter(1), reverse=True)
            for i in range(self.bad_candidates):
                self.pop.append(ratings[i][0])
        for a in self.pop:
            a.info['looks_like'] = count_looks_like(a, all_cand, self.comparator)
        self.all_cand = all_cand
        self.__calc_participation__()

    def update(self):
        """ The update method in Population will add to the end of
        the population, that can't be used here since we might have
        bad candidates that need to stay in the population, therefore
        just recalc the population every time. """
        self.pop = []
        self.__initialize_pop__()
        self._write_log()

    def get_one_candidate(self):
        """Returns one candidates at random."""
        if len(self.pop) < 1:
            self.update()
        if len(self.pop) < 1:
            return None
        t = self.rng.randint(len(self.pop))
        c = self.pop[t]
        return c.copy()

    def get_two_candidates(self):
        """Returns two candidates at random."""
        if len(self.pop) < 2:
            self.update()
        if len(self.pop) < 2:
            return None
        c1 = self.pop[0]
        c2 = self.pop[0]
        used_before = False
        while c1.info['confid'] == c2.info['confid'] and (not used_before):
            t = self.rng.randint(len(self.pop))
            c1 = self.pop[t]
            t = self.rng.randint(len(self.pop))
            c2 = self.pop[t]
            c1id = c1.info['confid']
            c2id = c2.info['confid']
            used_before = tuple(sorted([c1id, c2id])) in self.pairs and self.exclude_used_pairs
        return (c1.copy(), c2.copy())