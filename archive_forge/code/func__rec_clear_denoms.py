from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def _rec_clear_denoms(g, v, K0, K1):
    """Recursive helper for :func:`dmp_clear_denoms`."""
    common = K1.one
    if not v:
        for c in g:
            common = K1.lcm(common, K0.denom(c))
    else:
        w = v - 1
        for c in g:
            common = K1.lcm(common, _rec_clear_denoms(c, w, K0, K1))
    return common