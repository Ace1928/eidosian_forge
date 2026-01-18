import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _get_TV(codons1, codons2, codon_table):
    """Get TV (PRIVATE).

    Arguments:
     - T - proportions of transitional differences
     - V - proportions of transversional differences

    """
    purine = ('A', 'G')
    pyrimidine = ('C', 'T')
    TV = [0, 0]
    sites = 0
    for codon1, codon2 in zip(codons1, codons2):
        for nucleotide1, nucleotide2 in zip(codon1, codon2):
            if nucleotide1 == nucleotide2:
                pass
            elif nucleotide1 in purine and nucleotide2 in purine:
                TV[0] += 1
            elif nucleotide1 in pyrimidine and nucleotide2 in pyrimidine:
                TV[0] += 1
            else:
                TV[1] += 1
            sites += 1
    return (TV[0] / sites, TV[1] / sites)