import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _get_Q(pi, k, w, codons, codon_table):
    """Q matrix for codon substitution (PRIVATE)."""
    codon_num = len(codons)
    Q = np.zeros((codon_num, codon_num))
    for i1, codon1 in enumerate(codons):
        for i2, codon2 in enumerate(codons):
            if i1 != i2:
                Q[i1, i2] = _q(codon1, codon2, pi, k, w, codon_table=codon_table)
    nucl_substitutions = 0
    for i, codon in enumerate(codons):
        Q[i, i] = -sum(Q[i, :])
        try:
            nucl_substitutions += pi[codon] * -Q[i, i]
        except KeyError:
            pass
    Q /= nucl_substitutions
    return Q