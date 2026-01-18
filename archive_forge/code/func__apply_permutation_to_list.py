import bisect
from collections import defaultdict
from sympy.combinatorics import Permutation
from sympy.core.containers import Tuple
from sympy.core.numbers import Integer
def _apply_permutation_to_list(perm: Permutation, target_list: list):
    """
    Permute a list according to the given permutation.
    """
    new_list = [None for i in range(perm.size)]
    for i, e in enumerate(target_list):
        new_list[perm(i)] = e
    return new_list