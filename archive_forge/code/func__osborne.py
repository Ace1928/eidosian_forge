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
def _osborne(e, d):
    """Replace all hyperbolic functions with trig functions using
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
        if not isinstance(rv, HyperbolicFunction):
            return rv
        a = rv.args[0]
        a = a * d if not a.is_Add else Add._from_args([i * d for i in a.args])
        if isinstance(rv, sinh):
            return I * sin(a)
        elif isinstance(rv, cosh):
            return cos(a)
        elif isinstance(rv, tanh):
            return I * tan(a)
        elif isinstance(rv, coth):
            return cot(a) / I
        elif isinstance(rv, sech):
            return sec(a)
        elif isinstance(rv, csch):
            return csc(a) / I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)
    return bottom_up(e, f)