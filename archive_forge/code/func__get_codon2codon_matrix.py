import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _get_codon2codon_matrix(codon_table):
    """Get codon codon substitution matrix (PRIVATE).

    Elements in the matrix are number of synonymous and nonsynonymous
    substitutions required for the substitution.
    """
    bases = ('A', 'T', 'C', 'G')
    codons = [codon for codon in list(codon_table.forward_table.keys()) + codon_table.stop_codons if 'U' not in codon]
    codon_dict = codon_table.forward_table.copy()
    for stop in codon_table.stop_codons:
        codon_dict[stop] = 'stop'
    num = len(codons)
    G = {}
    nonsyn_G = {}
    graph = {}
    graph_nonsyn = {}
    for i, codon in enumerate(codons):
        graph[codon] = {}
        graph_nonsyn[codon] = {}
        for p in range(3):
            for base in bases:
                tmp_codon = codon[0:p] + base + codon[p + 1:]
                if codon_dict[codon] != codon_dict[tmp_codon]:
                    graph_nonsyn[codon][tmp_codon] = 1
                    graph[codon][tmp_codon] = 1
                elif codon != tmp_codon:
                    graph_nonsyn[codon][tmp_codon] = 0.1
                    graph[codon][tmp_codon] = 1
    for codon1 in codons:
        nonsyn_G[codon1] = {}
        G[codon1] = {}
        for codon2 in codons:
            if codon1 == codon2:
                nonsyn_G[codon1][codon2] = 0
                G[codon1][codon2] = 0
            else:
                nonsyn_G[codon1][codon2] = _dijkstra(graph_nonsyn, codon1, codon2)
                G[codon1][codon2] = _dijkstra(graph, codon1, codon2)
    return (G, nonsyn_G)