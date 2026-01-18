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
class HyperRep_log2(HyperRep):
    """ Represent log(1/2 + sqrt(1 - z)/2) == -z/4*hyper([3/2, 1, 1], [2, 2], z) """

    @classmethod
    def _expr_small(cls, z):
        return log(S.Half + sqrt(1 - z) / 2)

    @classmethod
    def _expr_small_minus(cls, z):
        return log(S.Half + sqrt(1 + z) / 2)

    @classmethod
    def _expr_big(cls, z, n):
        if n.is_even:
            return (n - S.Half) * pi * I + log(sqrt(z) / 2) + I * asin(1 / sqrt(z))
        else:
            return (n - S.Half) * pi * I + log(sqrt(z) / 2) - I * asin(1 / sqrt(z))

    def _expr_big_minus(cls, z, n):
        if n.is_even:
            return pi * I * n + log(S.Half + sqrt(1 + z) / 2)
        else:
            return pi * I * n + log(sqrt(1 + z) / 2 - S.Half)