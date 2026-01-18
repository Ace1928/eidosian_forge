import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def _count_diff_NG86(codon1, codon2, codon_table):
    """Count differences between two codons, three-letter string (PRIVATE).

    The function will take multiple pathways from codon1 to codon2
    into account.
    """
    SN = [0, 0]
    if codon1 == codon2:
        return SN
    else:
        diff_pos = [i for i, (nucleotide1, nucleotide2) in enumerate(zip(codon1, codon2)) if nucleotide1 != nucleotide2]

        def compare_codon(codon1, codon2, codon_table, weight=1):
            """Compare two codon accounting for different pathways."""
            sd = nd = 0
            if len(set(map(codon_table.forward_table.get, [codon1, codon2]))) == 1:
                sd += weight
            else:
                nd += weight
            return (sd, nd)
        if len(diff_pos) == 1:
            SN = [i + j for i, j in zip(SN, compare_codon(codon1, codon2, codon_table=codon_table))]
        elif len(diff_pos) == 2:
            for i in diff_pos:
                temp_codon = codon1[:i] + codon2[i] + codon1[i + 1:]
                SN = [i + j for i, j in zip(SN, compare_codon(codon1, temp_codon, codon_table=codon_table, weight=0.5))]
                SN = [i + j for i, j in zip(SN, compare_codon(temp_codon, codon2, codon_table=codon_table, weight=0.5))]
        elif len(diff_pos) == 3:
            paths = list(permutations([0, 1, 2], 3))
            tmp_codon = []
            for index1, index2, index3 in paths:
                tmp1 = codon1[:index1] + codon2[index1] + codon1[index1 + 1:]
                tmp2 = tmp1[:index2] + codon2[index2] + tmp1[index2 + 1:]
                tmp_codon.append((tmp1, tmp2))
                SN = [i + j for i, j in zip(SN, compare_codon(codon1, tmp1, codon_table, weight=0.5 / 3))]
                SN = [i + j for i, j in zip(SN, compare_codon(tmp1, tmp2, codon_table, weight=0.5 / 3))]
                SN = [i + j for i, j in zip(SN, compare_codon(tmp2, codon2, codon_table, weight=0.5 / 3))]
    return SN