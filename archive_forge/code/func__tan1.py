from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational, igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math
def _tan1(p, x, prec):
    """
    Helper function of :func:`rs_tan`.

    Return the series expansion of tan of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atan is
    easier than that of tan.

    Consider `f(x) = y - \\arctan(x)`
    Let r be a root of f(x) found using Newton's method.
    Then `f(r) = 0`
    Or `y = \\arctan(x)` where `x = \\tan(y)` as required.
    """
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atan(p1, x, precx)
        tmp = rs_mul(tmp, 1 + rs_square(p1, x, precx), x, precx)
        p1 += tmp
    return p1