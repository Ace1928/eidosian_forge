from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def _rec_eval_in(g, a, v, i, j, K):
    """Recursive helper for :func:`dmp_eval_in`."""
    if i == j:
        return dmp_eval(g, a, v, K)
    v, i = (v - 1, i + 1)
    return dmp_strip([_rec_eval_in(c, a, v, i, j, K) for c in g], v)