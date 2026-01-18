import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _q(codon1, codon2, pi, k, w, codon_table):
    """Q matrix for codon substitution (PRIVATE).

    Arguments:
     - codon1, codon2  : three letter codon string
     - pi              : expected codon frequency
     - k               : transition/transversion ratio
     - w               : nonsynonymous/synonymous rate ratio
     - codon_table     : Bio.Data.CodonTable object

    """
    if codon1 == codon2:
        return 0
    if codon1 in codon_table.stop_codons or codon2 in codon_table.stop_codons:
        return 0
    if codon1 not in pi or codon2 not in pi:
        return 0
    purine = ('A', 'G')
    pyrimidine = ('T', 'C')
    diff = [(i, nucleotide1, nucleotide2) for i, (nucleotide1, nucleotide2) in enumerate(zip(codon1, codon2)) if nucleotide1 != nucleotide2]
    if len(diff) >= 2:
        return 0
    if codon_table.forward_table[codon1] == codon_table.forward_table[codon2]:
        if diff[0][1] in purine and diff[0][2] in purine:
            return k * pi[codon2]
        elif diff[0][1] in pyrimidine and diff[0][2] in pyrimidine:
            return k * pi[codon2]
        else:
            return pi[codon2]
    elif diff[0][1] in purine and diff[0][2] in purine:
        return w * k * pi[codon2]
    elif diff[0][1] in pyrimidine and diff[0][2] in pyrimidine:
        return w * k * pi[codon2]
    else:
        return w * pi[codon2]