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
def TR22(rv, max=4, pow=False):
    """Convert tan(x)**2 to sec(x)**2 - 1 and cot(x)**2 to csc(x)**2 - 1.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR22
    >>> from sympy.abc import x
    >>> from sympy import tan, cot
    >>> TR22(1 + tan(x)**2)
    sec(x)**2
    >>> TR22(1 + cot(x)**2)
    csc(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and rv.base.func in (cot, tan)):
            return rv
        rv = _TR56(rv, tan, sec, lambda x: x - 1, max=max, pow=pow)
        rv = _TR56(rv, cot, csc, lambda x: x - 1, max=max, pow=pow)
        return rv
    return bottom_up(rv, f)