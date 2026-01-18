from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def _rec_integrate_in(g, m, v, i, j, K):
    """Recursive helper for :func:`dmp_integrate_in`."""
    if i == j:
        return dmp_integrate(g, m, v, K)
    w, i = (v - 1, i + 1)
    return dmp_strip([_rec_integrate_in(c, m, w, i, j, K) for c in g], v)