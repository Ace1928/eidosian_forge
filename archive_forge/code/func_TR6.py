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
def TR6(rv, max=4, pow=False):
    """Replacement of cos**2 with 1 - sin(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR6
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR6(cos(x)**2)
    1 - sin(x)**2
    >>> TR6(cos(x)**-2)  #unchanged
    cos(x)**(-2)
    >>> TR6(cos(x)**4)
    (1 - sin(x)**2)**2
    """
    return _TR56(rv, cos, sin, lambda x: 1 - x, max=max, pow=pow)