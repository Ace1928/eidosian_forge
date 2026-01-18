from __future__ import annotations
from sympy.core.function import Function
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity, perfect_power
from sympy.utilities.misc import as_int
from sympy.core.random import _randint, randint
from itertools import cycle, product
def _valid_expr(expr):
    """
    return coefficients of expr if it is a univariate polynomial
    with integer coefficients else raise a ValueError.
    """
    if not expr.is_polynomial():
        raise ValueError('The expression should be a polynomial')
    polynomial = Poly(expr)
    if not polynomial.is_univariate:
        raise ValueError('The expression should be univariate')
    if not polynomial.domain == ZZ:
        raise ValueError('The expression should should have integer coefficients')
    return polynomial.all_coeffs()