from sympy.calculus.accumulationbounds import AccumulationBounds
from sympy.core.add import Add
from sympy.core.function import PoleError
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import factorial, subfactorial
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series.limits import Limit
def _limit_seq(expr, n, trials):
    from sympy.concrete.summations import Sum
    for i in range(trials):
        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result
        num, den = expr.as_numer_denom()
        if not den.has(n) or not num.has(n):
            result = _limit_inf(expr.doit(), n)
            if result is not None:
                return result
            return None
        num, den = (difference_delta(t.expand(), n) for t in [num, den])
        expr = (num / den).gammasimp()
        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result
        num, den = expr.as_numer_denom()
        num = dominant(num, n)
        if num is None:
            return None
        den = dominant(den, n)
        if den is None:
            return None
        expr = (num / den).gammasimp()