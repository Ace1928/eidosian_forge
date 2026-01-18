from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def _dup_left_decompose(f, h, K):
    """Helper function for :func:`_dup_decompose`."""
    g, i = ({}, 0)
    while f:
        q, r = dup_div(f, h, K)
        if dup_degree(r) > 0:
            return None
        else:
            g[i] = dup_LC(r, K)
            f, i = (q, i + 1)
    return dup_from_raw_dict(g, K)