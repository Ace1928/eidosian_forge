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
def _imag_count_of_factor(f):
    """Return the number of imaginary roots for irreducible
    univariate polynomial ``f``.
    """
    terms = [(i, j) for (i,), j in f.terms()]
    if any((i % 2 for i, j in terms)):
        return 0
    even = [(i, I ** i * j) for i, j in terms]
    even = Poly.from_dict(dict(even), Dummy('x'))
    return int(even.count_roots(-oo, oo))