from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
from sympy.functions.elementary.trigonometric import (
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG
def _osbornei(e, d):
    """Replace all trig functions with hyperbolic functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """

    def f(rv):
        if not isinstance(rv, TrigonometricFunction):
            return rv
        const, x = rv.args[0].as_independent(d, as_Add=True)
        a = x.xreplace({d: S.One}) + const * I
        if isinstance(rv, sin):
            return sinh(a) / I
        elif isinstance(rv, cos):
            return cosh(a)
        elif isinstance(rv, tan):
            return tanh(a) / I
        elif isinstance(rv, cot):
            return coth(a) * I
        elif isinstance(rv, sec):
            return sech(a)
        elif isinstance(rv, csc):
            return csch(a) * I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)
    return bottom_up(e, f)