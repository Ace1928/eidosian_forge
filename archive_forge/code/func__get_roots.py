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
def _get_roots(cls, method, poly, radicals):
    """Return postprocessed roots of specified kind. """
    if not poly.is_univariate:
        raise PolynomialError('only univariate polynomials are allowed')
    d = Dummy()
    poly = poly.subs(poly.gen, d)
    x = symbols('x')
    free_names = {str(i) for i in poly.free_symbols}
    for x in chain((symbols('x'),), numbered_symbols('x')):
        if x.name not in free_names:
            poly = poly.xreplace({d: x})
            break
    coeff, poly = cls._preprocess_roots(poly)
    roots = []
    for root in getattr(cls, method)(poly):
        roots.append(coeff * cls._postprocess_root(root, radicals))
    return roots