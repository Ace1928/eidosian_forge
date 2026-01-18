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
class HyperRep_asin2(HyperRep):
    """ Represent hyper([1, 1], [3/2], z) == asin(sqrt(z))/sqrt(z)/sqrt(1-z). """

    @classmethod
    def _expr_small(cls, z):
        return HyperRep_asin1._expr_small(z) / HyperRep_power1._expr_small(S.Half, z)

    @classmethod
    def _expr_small_minus(cls, z):
        return HyperRep_asin1._expr_small_minus(z) / HyperRep_power1._expr_small_minus(S.Half, z)

    @classmethod
    def _expr_big(cls, z, n):
        return HyperRep_asin1._expr_big(z, n) / HyperRep_power1._expr_big(S.Half, z, n)

    @classmethod
    def _expr_big_minus(cls, z, n):
        return HyperRep_asin1._expr_big_minus(z, n) / HyperRep_power1._expr_big_minus(S.Half, z, n)