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
class HyperRep_sqrts2(HyperRep):
    """ Return a representative for
          sqrt(z)/2*[(1-sqrt(z))**2a - (1 + sqrt(z))**2a]
          == -2*z/(2*a+1) d/dz hyper([-a - 1/2, -a], [1/2], z)"""

    @classmethod
    def _expr_small(cls, a, z):
        return sqrt(z) * ((1 - sqrt(z)) ** (2 * a) - (1 + sqrt(z)) ** (2 * a)) / 2

    @classmethod
    def _expr_small_minus(cls, a, z):
        return sqrt(z) * (1 + z) ** a * sin(2 * a * atan(sqrt(z)))

    @classmethod
    def _expr_big(cls, a, z, n):
        if n.is_even:
            return sqrt(z) / 2 * ((sqrt(z) - 1) ** (2 * a) * exp(2 * pi * I * a * (n - 1)) - (sqrt(z) + 1) ** (2 * a) * exp(2 * pi * I * a * n))
        else:
            n -= 1
            return sqrt(z) / 2 * ((sqrt(z) - 1) ** (2 * a) * exp(2 * pi * I * a * (n + 1)) - (sqrt(z) + 1) ** (2 * a) * exp(2 * pi * I * a * n))

    def _expr_big_minus(cls, a, z, n):
        if n.is_even:
            return (1 + z) ** a * exp(2 * pi * I * n * a) * sqrt(z) * sin(2 * a * atan(sqrt(z)))
        else:
            return (1 + z) ** a * exp(2 * pi * I * n * a) * sqrt(z) * sin(2 * a * atan(sqrt(z)) - 2 * pi * a)