import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _count_replacement(codons, G):
    """Count replacement needed for a given codon_set (PRIVATE)."""
    if len(codons) == 1:
        return (0, 0)
    elif len(codons) == 2:
        codons = list(codons)
        return floor(G[codons[0]][codons[1]])
    else:
        subgraph = {codon1: {codon2: G[codon1][codon2] for codon2 in codons if codon1 != codon2} for codon1 in codons}
        return _prim(subgraph)