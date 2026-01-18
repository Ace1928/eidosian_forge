import numpy as np
from ase.ga import get_raw_score
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