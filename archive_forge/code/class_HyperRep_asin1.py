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
class HyperRep_asin1(HyperRep):
    """ Represent hyper([1/2, 1/2], [3/2], z) == asin(sqrt(z))/sqrt(z). """

    @classmethod
    def _expr_small(cls, z):
        return asin(sqrt(z)) / sqrt(z)

    @classmethod
    def _expr_small_minus(cls, z):
        return asinh(sqrt(z)) / sqrt(z)

    @classmethod
    def _expr_big(cls, z, n):
        return S.NegativeOne ** n * ((S.Half - n) * pi / sqrt(z) + I * acosh(sqrt(z)) / sqrt(z))

    @classmethod
    def _expr_big_minus(cls, z, n):
        return S.NegativeOne ** n * (asinh(sqrt(z)) / sqrt(z) + n * pi * I / sqrt(z))