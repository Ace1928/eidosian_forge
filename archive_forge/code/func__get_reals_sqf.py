from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
from sympy.utilities import lambdify, public, sift, numbered_symbols
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain
@classmethod
def _get_reals_sqf(cls, currentfactor, use_cache=True):
    """Get real root isolating intervals for a square-free factor."""
    if use_cache and currentfactor in _reals_cache:
        real_part = _reals_cache[currentfactor]
    else:
        _reals_cache[currentfactor] = real_part = dup_isolate_real_roots_sqf(currentfactor.rep.rep, currentfactor.rep.dom, blackbox=True)
    return real_part