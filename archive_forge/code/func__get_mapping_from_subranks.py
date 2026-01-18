import bisect
from collections import defaultdict
from sympy.combinatorics import Permutation
from sympy.core.containers import Tuple
from sympy.core.numbers import Integer
def _get_mapping_from_subranks(subranks):
    mapping = {}
    counter = 0
    for i, rank in enumerate(subranks):
        for j in range(rank):
            mapping[counter] = (i, j)
            counter += 1
    return mapping