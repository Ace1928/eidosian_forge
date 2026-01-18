import bisect
from collections import defaultdict
from sympy.combinatorics import Permutation
from sympy.core.containers import Tuple
from sympy.core.numbers import Integer
def _build_push_indices_up_func_transformation(flattened_contraction_indices):
    shifts = {0: 0}
    i = 0
    cumulative = 0
    while i < len(flattened_contraction_indices):
        j = 1
        while i + j < len(flattened_contraction_indices):
            if flattened_contraction_indices[i] + j != flattened_contraction_indices[i + j]:
                break
            j += 1
        cumulative += j
        shifts[flattened_contraction_indices[i]] = cumulative
        i += j
    shift_keys = sorted(shifts.keys())

    def func(idx):
        return shifts[shift_keys[bisect.bisect_right(shift_keys, idx) - 1]]

    def transform(j):
        if j in flattened_contraction_indices:
            return None
        else:
            return j - func(j)
    return transform