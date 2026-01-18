from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.galoistools import (
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
def _dup_rr_trivial_gcd(f, g, K):
    """Handle trivial cases in GCD algorithm over a ring. """
    if not (f or g):
        return ([], [], [])
    elif not f:
        if K.is_nonnegative(dup_LC(g, K)):
            return (g, [], [K.one])
        else:
            return (dup_neg(g, K), [], [-K.one])
    elif not g:
        if K.is_nonnegative(dup_LC(f, K)):
            return (f, [K.one], [])
        else:
            return (dup_neg(f, K), [-K.one], [])
    return None