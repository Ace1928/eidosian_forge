import bisect
from collections import defaultdict
from sympy.combinatorics import Permutation
from sympy.core.containers import Tuple
from sympy.core.numbers import Integer
def _get_argindex(subindices, ind):
    for i, sind in enumerate(subindices):
        if ind == sind:
            return i
        if isinstance(sind, (set, frozenset)) and ind in sind:
            return i
    raise IndexError('%s not found in %s' % (ind, subindices))