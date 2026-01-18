import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _G_test(site_counts):
    """G test for 2x2 contingency table (PRIVATE).

    Arguments:
     - site_counts - [syn_fix, nonsyn_fix, syn_poly, nonsyn_poly]

    >>> print("%0.6f" % _G_test([17, 7, 42, 2]))
    0.004924
    """
    G = 0
    tot = sum(site_counts)
    tot_syn = site_counts[0] + site_counts[2]
    tot_non = site_counts[1] + site_counts[3]
    tot_fix = sum(site_counts[:2])
    tot_poly = sum(site_counts[2:])
    exp = [tot_fix * tot_syn / tot, tot_fix * tot_non / tot, tot_poly * tot_syn / tot, tot_poly * tot_non / tot]
    for obs, ex in zip(site_counts, exp):
        G += obs * log(obs / ex)
    return erfc(sqrt(G))