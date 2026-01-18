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
def TR1(rv):
    """Replace sec, csc with 1/cos, 1/sin

    Examples
    ========

    >>> from sympy.simplify.fu import TR1, sec, csc
    >>> from sympy.abc import x
    >>> TR1(2*csc(x) + sec(x))
    1/cos(x) + 2/sin(x)
    """

    def f(rv):
        if isinstance(rv, sec):
            a = rv.args[0]
            return S.One / cos(a)
        elif isinstance(rv, csc):
            a = rv.args[0]
            return S.One / sin(a)
        return rv
    return bottom_up(rv, f)