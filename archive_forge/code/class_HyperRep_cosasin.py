from functools import reduce
from sympy.core import S, ilcm, Mod
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative, ArgumentIndexError
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import I, pi, oo, zoo
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import (sqrt, exp, log, sin, cos, asin, atan,
from sympy.functions import factorial, RisingFactorial
from sympy.functions.elementary.complexes import Abs, re, unpolarify
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
class HyperRep_cosasin(HyperRep):
    """ Represent hyper([a, -a], [1/2], z) == cos(2*a*asin(sqrt(z))). """

    @classmethod
    def _expr_small(cls, a, z):
        return cos(2 * a * asin(sqrt(z)))

    @classmethod
    def _expr_small_minus(cls, a, z):
        return cosh(2 * a * asinh(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        return cosh(2 * a * acosh(sqrt(z)) + a * pi * I * (2 * n - 1))

    @classmethod
    def _expr_big_minus(cls, a, z, n):
        return cosh(2 * a * asinh(sqrt(z)) + 2 * a * pi * I * n)