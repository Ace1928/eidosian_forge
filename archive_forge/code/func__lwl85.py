import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _lwl85(codons1, codons2, codon_table):
    """LWL85 method main function (PRIVATE).

    Nomenclature is according to Li et al. (1985), PMID 3916709.
    """
    codon_fold_dict = _get_codon_fold(codon_table)
    fold0 = [0, 0]
    fold2 = [0, 0]
    fold4 = [0, 0]
    for codon in codons1 + codons2:
        fold_num = codon_fold_dict[codon]
        for f in fold_num:
            if f == '0':
                fold0[0] += 1
            elif f == '2':
                fold2[0] += 1
            elif f == '4':
                fold4[0] += 1
    L = [sum(fold0) / 2.0, sum(fold2) / 2.0, sum(fold4) / 2.0]
    PQ = [0] * 6
    for codon1, codon2 in zip(codons1, codons2):
        if codon1 == codon2:
            continue
        PQ = [i + j for i, j in zip(PQ, _diff_codon(codon1, codon2, fold_dict=codon_fold_dict))]
    PQ = [i / j for i, j in zip(PQ, L * 2)]
    P = PQ[:3]
    Q = PQ[3:]
    A = [1.0 / 2 * log(1.0 / (1 - 2 * i - j)) - 1.0 / 4 * log(1.0 / (1 - 2 * j)) for i, j in zip(P, Q)]
    B = [1.0 / 2 * log(1.0 / (1 - 2 * i)) for i in Q]
    dS = 3 * (L[2] * A[1] + L[2] * (A[2] + B[2])) / (L[1] + 3 * L[2])
    dN = 3 * (L[2] * B[1] + L[0] * (A[0] + B[0])) / (2 * L[1] + 3 * L[0])
    return (dN, dS)