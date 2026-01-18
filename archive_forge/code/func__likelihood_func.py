import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _likelihood_func(t, k, w, pi, codon_cnt, codons, codon_table):
    """Likelihood function for ML method (PRIVATE)."""
    from scipy.linalg import expm
    Q = _get_Q(pi, k, w, codons, codon_table)
    P = expm(Q * t)
    likelihood = 0
    for i, codon1 in enumerate(codons):
        for j, codon2 in enumerate(codons):
            if (codon1, codon2) in codon_cnt:
                if P[i, j] * pi[codon1] <= 0:
                    likelihood += codon_cnt[codon1, codon2] * 0
                else:
                    likelihood += codon_cnt[codon1, codon2] * log(pi[codon1] * P[i, j])
    return likelihood