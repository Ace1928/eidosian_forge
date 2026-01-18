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
def TR9(rv):
    """Sum of ``cos`` or ``sin`` terms as a product of ``cos`` or ``sin``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR9
    >>> from sympy import cos, sin
    >>> TR9(cos(1) + cos(2))
    2*cos(1/2)*cos(3/2)
    >>> TR9(cos(1) + 2*sin(1) + 2*sin(2))
    cos(1) + 4*sin(3/2)*cos(1/2)

    If no change is made by TR9, no re-arrangement of the
    expression will be made. For example, though factoring
    of common term is attempted, if the factored expression
    was not changed, the original expression will be returned:

    >>> TR9(cos(3) + cos(3)*cos(2))
    cos(3) + cos(2)*cos(3)

    """

    def f(rv):
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            if not rv.is_Add:
                return rv
            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new
                            args[j] = None
                            hit = True
                            break
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)
                return rv
            split = trig_split(*args)
            if not split:
                return rv
            gcd, n1, n2, a, b, iscos = split
            if iscos:
                if n1 == n2:
                    return gcd * n1 * 2 * cos((a + b) / 2) * cos((a - b) / 2)
                if n1 < 0:
                    a, b = (b, a)
                return -2 * gcd * sin((a + b) / 2) * sin((a - b) / 2)
            else:
                if n1 == n2:
                    return gcd * n1 * 2 * sin((a + b) / 2) * cos((a - b) / 2)
                if n1 < 0:
                    a, b = (b, a)
                return 2 * gcd * cos((a + b) / 2) * sin((a - b) / 2)
        return process_common_addends(rv, do)
    return bottom_up(rv, f)