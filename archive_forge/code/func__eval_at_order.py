from sympy.core import Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial, factorial, RisingFactorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sec
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import (chebyshevt_poly, chebyshevu_poly,
@classmethod
def _eval_at_order(cls, n, m):
    P = legendre_poly(n, _x, polys=True).diff((_x, m))
    return S.NegativeOne ** m * (1 - _x ** 2) ** Rational(m, 2) * P.as_expr()