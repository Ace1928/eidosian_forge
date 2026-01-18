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
def _TR56(rv, f, g, h, max, pow):
    """Helper for TR5 and TR6 to replace f**2 with h(g**2)

    Options
    =======

    max :   controls size of exponent that can appear on f
            e.g. if max=4 then f**4 will be changed to h(g**2)**2.
    pow :   controls whether the exponent must be a perfect power of 2
            e.g. if pow=True (and max >= 6) then f**6 will not be changed
            but f**8 will be changed to h(g**2)**4

    >>> from sympy.simplify.fu import _TR56 as T
    >>> from sympy.abc import x
    >>> from sympy import sin, cos
    >>> h = lambda x: 1 - x
    >>> T(sin(x)**3, sin, cos, h, 4, False)
    (1 - cos(x)**2)*sin(x)
    >>> T(sin(x)**6, sin, cos, h, 6, False)
    (1 - cos(x)**2)**3
    >>> T(sin(x)**6, sin, cos, h, 6, True)
    sin(x)**6
    >>> T(sin(x)**8, sin, cos, h, 10, True)
    (1 - cos(x)**2)**4
    """

    def _f(rv):
        if not (rv.is_Pow and rv.base.func == f):
            return rv
        if not rv.exp.is_real:
            return rv
        if (rv.exp < 0) == True:
            return rv
        if (rv.exp > max) == True:
            return rv
        if rv.exp == 1:
            return rv
        if rv.exp == 2:
            return h(g(rv.base.args[0]) ** 2)
        else:
            if rv.exp % 2 == 1:
                e = rv.exp // 2
                return f(rv.base.args[0]) * h(g(rv.base.args[0]) ** 2) ** e
            elif rv.exp == 4:
                e = 2
            elif not pow:
                if rv.exp % 2:
                    return rv
                e = rv.exp // 2
            else:
                p = perfect_power(rv.exp)
                if not p:
                    return rv
                e = rv.exp // 2
            return h(g(rv.base.args[0]) ** 2) ** e
    return bottom_up(rv, _f)