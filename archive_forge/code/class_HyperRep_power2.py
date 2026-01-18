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
class HyperRep_power2(HyperRep):
    """ Return a representative for hyper([a, a - 1/2], [2*a], z). """

    @classmethod
    def _expr_small(cls, a, x):
        return 2 ** (2 * a - 1) * (1 + sqrt(1 - x)) ** (1 - 2 * a)

    @classmethod
    def _expr_small_minus(cls, a, x):
        return 2 ** (2 * a - 1) * (1 + sqrt(1 + x)) ** (1 - 2 * a)

    @classmethod
    def _expr_big(cls, a, x, n):
        sgn = -1
        if n.is_odd:
            sgn = 1
            n -= 1
        return 2 ** (2 * a - 1) * (1 + sgn * I * sqrt(x - 1)) ** (1 - 2 * a) * exp(-2 * n * pi * I * a)

    @classmethod
    def _expr_big_minus(cls, a, x, n):
        sgn = 1
        if n.is_odd:
            sgn = -1
        return sgn * 2 ** (2 * a - 1) * (sqrt(1 + x) + sgn) ** (1 - 2 * a) * exp(-2 * pi * I * a * n)