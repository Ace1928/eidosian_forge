import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _get_codon_fold(codon_table):
    """Classify different position in a codon into different folds (PRIVATE)."""
    fold_table = {}
    forward_table = codon_table.forward_table
    bases = {'A', 'T', 'C', 'G'}
    for codon in forward_table:
        if 'U' in codon:
            continue
        fold = ''
        codon_base_lst = list(codon)
        for i, base in enumerate(codon_base_lst):
            other_bases = bases - set(base)
            aa = []
            for other_base in other_bases:
                codon_base_lst[i] = other_base
                try:
                    aa.append(forward_table[''.join(codon_base_lst)])
                except KeyError:
                    aa.append('stop')
            if aa.count(forward_table[codon]) == 0:
                fold += '0'
            elif aa.count(forward_table[codon]) in (1, 2):
                fold += '2'
            elif aa.count(forward_table[codon]) == 3:
                fold += '4'
            else:
                raise RuntimeError('Unknown Error, cannot assign the position to a fold')
            codon_base_lst[i] = base
        fold_table[codon] = fold
    return fold_table