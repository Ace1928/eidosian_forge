import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _diff_codon(codon1, codon2, fold_dict):
    """Count number of different substitution types between two codons (PRIVATE).

    returns tuple (P0, P2, P4, Q0, Q2, Q4)

    Nomenclature is according to Li et al. (1958), PMID 3916709.
    """
    P0 = P2 = P4 = Q0 = Q2 = Q4 = 0
    fold_num = fold_dict[codon1]
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    for n, (nucleotide1, nucleotide2) in enumerate(zip(codon1, codon2)):
        if nucleotide1 == nucleotide2:
            pass
        elif nucleotide1 in purine and nucleotide2 in purine:
            if fold_num[n] == '0':
                P0 += 1
            elif fold_num[n] == '2':
                P2 += 1
            elif fold_num[n] == '4':
                P4 += 1
            else:
                raise RuntimeError('Unexpected fold_num %d' % fold_num[n])
        elif nucleotide1 in pyrimidine and nucleotide2 in pyrimidine:
            if fold_num[n] == '0':
                P0 += 1
            elif fold_num[n] == '2':
                P2 += 1
            elif fold_num[n] == '4':
                P4 += 1
            else:
                raise RuntimeError('Unexpected fold_num %d' % fold_num[n])
        elif fold_num[n] == '0':
            Q0 += 1
        elif fold_num[n] == '2':
            Q2 += 1
        elif fold_num[n] == '4':
            Q4 += 1
        else:
            raise RuntimeError('Unexpected fold_num %d' % fold_num[n])
    return (P0, P2, P4, Q0, Q2, Q4)