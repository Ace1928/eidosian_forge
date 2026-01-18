from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
class MultiObjectivePopulation(RankFitnessPopulation):
    """ Allows for assignment of fitness based on a set of two variables
        such that fitness is ranked according to a Pareto-front of
        non-dominated candidates.

    Parameters
    ----------
        abs_data: list
            Set of key_value_pairs in atoms object for which fitness should
            be assigned based on absolute value.

        rank_data: list
            Set of key_value_pairs in atoms object for which data should
            be ranked in order to ascribe fitness.

        variable_function: function
            A function that takes as input an Atoms object and returns
            the variable that differentiates the ranks. Only use if
            data is ranked.

        exp_function: boolean
            If True use an exponential function for ranking the fitness.
            If False use the same as in Population. Default True.

        exp_prefactor: float
            The prefactor used in the exponential fitness scaling function.
            Default 0.5

    """

    def __init__(self, data_connection, population_size, variable_function=None, comparator=None, logfile=None, use_extinct=False, abs_data=None, rank_data=None, exp_function=True, exp_prefactor=0.5):
        self.current_fitness = None
        if rank_data is None:
            rank_data = []
        self.rank_data = rank_data
        if abs_data is None:
            abs_data = []
        self.abs_data = abs_data
        RankFitnessPopulation.__init__(self, data_connection, population_size, variable_function, comparator, logfile, use_extinct, exp_function, exp_prefactor)

    def get_nonrank(self, nrcand, key=None):
        """"Returns a list of fitness values."""
        nrc_list = []
        for nrc in nrcand:
            nrc_list.append(nrc.info['key_value_pairs'][key])
        return nrc_list

    def __get_fitness__(self, candidates):
        msg = 'This is a multi-objective fitness function'
        msg += ' so there must be at least two datasets'
        msg += ' stated in the rank_data and abs_data variables'
        assert len(self.rank_data) + len(self.abs_data) >= 2, msg
        expf = self.exp_function
        all_fitnesses = []
        used = set()
        for rd in self.rank_data:
            used.add(rd)
            all_fitnesses.append(self.get_rank(candidates, key=rd))
        for d in self.abs_data:
            if d not in used:
                used.add(d)
                all_fitnesses.append(self.get_nonrank(candidates, key=d))
        fordered = list(zip(range(len(all_fitnesses[0])), *all_fitnesses))
        mvf_rank = -1
        rec_vrc = []
        mvf_list = []
        fordered.sort(key=itemgetter(1), reverse=True)
        for a in fordered:
            order, rest = (a[0], a[1:])
            if order not in rec_vrc:
                pff = []
                pff2 = []
                rec_vrc.append(order)
                pff.append((order, rest))
                for b in fordered:
                    border, brest = (b[0], b[1:])
                    if border not in rec_vrc:
                        if np.any(np.array(brest) >= np.array(rest)):
                            pff.append((border, brest))
                for na in pff:
                    norder, nrest = (na[0], na[1:])
                    dom = False
                    for nb in pff:
                        nborder, nbrest = (nb[0], nb[1:])
                        if norder != nborder:
                            if np.all(np.array(nbrest) > np.array(nrest)):
                                dom = True
                    if not dom:
                        pff2.append((norder, nrest))
                for ffa in pff2:
                    fforder, ffrest = (ffa[0], ffa[1:])
                    rec_vrc.append(fforder)
                    mvf_list.append((fforder, mvf_rank, ffrest))
                mvf_rank = mvf_rank - 1
        mvf_list.sort(key=itemgetter(0), reverse=False)
        rfro = np.array(list(zip(*mvf_list))[1])
        if not expf:
            rmax = max(rfro)
            rmin = min(rfro)
            T = rmin - rmax
            msg = 'Equal fitness for best and worst candidate in the '
            msg += 'population! Fitness scaling is impossible! '
            msg += 'Try with a larger population.'
            assert T != 0.0, msg
            return 0.5 * (1.0 - np.tanh(2.0 * (rfro - rmax) / T - 1.0))
        else:
            return self.exp_prefactor ** (-rfro - 1)

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
                if self.vf is not None:
                    c_vf = self.vf(c)
                i += 1
                eq = False
                for a in self.pop:
                    if self.vf is not None:
                        a_vf = self.vf(a)
                        if a_vf == c_vf:
                            if self.comparator.looks_like(a, c):
                                eq = True
                                break
                    elif self.comparator.looks_like(a, c):
                        eq = True
                        break
                if not eq:
                    self.pop.append(c)
        self.all_cand = all_cand