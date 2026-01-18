import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _get_pi(codons1, codons2, cmethod, codon_table):
    """Obtain codon frequency dict (pi) from two codon list (PRIVATE).

    This function is designed for ML method. Available counting methods
    (cfreq) are F1x4, F3x4 and F64.
    """
    pi = {}
    if cmethod == 'F1x4':
        fcodon = Counter((nucleotide for codon in codons1 + codons2 for nucleotide in codon))
        tot = sum(fcodon.values())
        fcodon = {j: k / tot for j, k in fcodon.items()}
        for codon in codon_table.forward_table.keys() + codon_table.stop_codons:
            if 'U' not in codon:
                pi[codon] = fcodon[codon[0]] * fcodon[codon[1]] * fcodon[codon[2]]
    elif cmethod == 'F3x4':
        fcodon = [{'A': 0, 'G': 0, 'C': 0, 'T': 0}, {'A': 0, 'G': 0, 'C': 0, 'T': 0}, {'A': 0, 'G': 0, 'C': 0, 'T': 0}]
        for codon in codons1 + codons2:
            fcodon[0][codon[0]] += 1
            fcodon[1][codon[1]] += 1
            fcodon[2][codon[2]] += 1
        for i in range(3):
            tot = sum(fcodon[i].values())
            fcodon[i] = {j: k / tot for j, k in fcodon[i].items()}
        for codon in list(codon_table.forward_table.keys()) + codon_table.stop_codons:
            if 'U' not in codon:
                pi[codon] = fcodon[0][codon[0]] * fcodon[1][codon[1]] * fcodon[2][codon[2]]
    elif cmethod == 'F61':
        for codon in codon_table.forward_table.keys() + codon_table.stop_codons:
            if 'U' not in codon:
                pi[codon] = 0.1
        for codon in codons1 + codons2:
            pi[codon] += 1
        tot = sum(pi.values())
        pi = {j: k / tot for j, k in pi.items()}
    return pi