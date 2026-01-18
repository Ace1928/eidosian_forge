from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from mpmath.libmp.libmpf import prec_to_dps
def _check_norm(elements, norm):
    """validate if input norm is consistent"""
    if norm is not None and norm.is_number:
        if norm.is_positive is False:
            raise ValueError('Input norm must be positive.')
        numerical = all((i.is_number and i.is_real is True for i in elements))
        if numerical and is_eq(norm ** 2, sum((i ** 2 for i in elements))) is False:
            raise ValueError('Incompatible value for norm.')