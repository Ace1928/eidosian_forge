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
def _preprocess_roots(cls, poly):
    """Take heroic measures to make ``poly`` compatible with ``CRootOf``."""
    dom = poly.get_domain()
    if not dom.is_Exact:
        poly = poly.to_exact()
    coeff, poly = preprocess_roots(poly)
    dom = poly.get_domain()
    if not dom.is_ZZ:
        raise NotImplementedError('sorted roots not supported over %s' % dom)
    return (coeff, poly)