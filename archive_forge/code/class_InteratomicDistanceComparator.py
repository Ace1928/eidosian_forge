import numpy as np
from ase.ga import get_raw_score
class InteratomicDistanceComparator:
    """ An implementation of the comparison criteria described in
          L.B. Vilhelmsen and B. Hammer, PRL, 108, 126101 (2012)

        Parameters:

        n_top: The number of atoms being optimized by the GA.
            Default 0 - meaning all atoms.

        pair_cor_cum_diff: The limit in eq. 2 of the letter.
        pair_cor_max: The limit in eq. 3 of the letter
        dE: The limit of eq. 1 of the letter
        mic: Determines if distances are calculated
        using the minimum image convention
    """

    def __init__(self, n_top=None, pair_cor_cum_diff=0.015, pair_cor_max=0.7, dE=0.02, mic=False):
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
        self.n_top = n_top or 0
        self.mic = mic

    def looks_like(self, a1, a2):
        """ Return if structure a1 or a2 are similar or not. """
        if len(a1) != len(a2):
            raise Exception('The two configurations are not the same size')
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
        if dE >= self.dE:
            return False
        a1top = a1[-self.n_top:]
        a2top = a2[-self.n_top:]
        cum_diff, max_diff = self.__compare_structure__(a1top, a2top)
        return cum_diff < self.pair_cor_cum_diff and max_diff < self.pair_cor_max

    def __compare_structure__(self, a1, a2):
        """ Private method for calculating the structural difference. """
        p1 = get_sorted_dist_list(a1, mic=self.mic)
        p2 = get_sorted_dist_list(a2, mic=self.mic)
        numbers = a1.numbers
        total_cum_diff = 0.0
        max_diff = 0
        for n in p1.keys():
            cum_diff = 0.0
            c1 = p1[n]
            c2 = p2[n]
            assert len(c1) == len(c2)
            if len(c1) == 0:
                continue
            t_size = np.sum(c1)
            d = np.abs(c1 - c2)
            cum_diff = np.sum(d)
            max_diff = np.max(d)
            ntype = float(sum([i == n for i in numbers]))
            total_cum_diff += cum_diff / t_size * ntype / float(len(numbers))
        return (total_cum_diff, max_diff)