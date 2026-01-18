from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def _dup_right_decompose(f, s, K):
    """Helper function for :func:`_dup_decompose`."""
    n = len(f) - 1
    lc = dup_LC(f, K)
    f = dup_to_raw_dict(f)
    g = {s: K.one}
    r = n // s
    for i in range(1, s):
        coeff = K.zero
        for j in range(0, i):
            if not n + j - i in f:
                continue
            if not s - j in g:
                continue
            fc, gc = (f[n + j - i], g[s - j])
            coeff += (i - r * j) * fc * gc
        g[s - i] = K.quo(coeff, i * r * lc)
    return dup_from_raw_dict(g, K)