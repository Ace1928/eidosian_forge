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
def TR8(rv, first=True):
    """Converting products of ``cos`` and/or ``sin`` to a sum or
    difference of ``cos`` and or ``sin`` terms.

    Examples
    ========

    >>> from sympy.simplify.fu import TR8
    >>> from sympy import cos, sin
    >>> TR8(cos(2)*cos(3))
    cos(5)/2 + cos(1)/2
    >>> TR8(cos(2)*sin(3))
    sin(5)/2 + sin(1)/2
    >>> TR8(sin(2)*sin(3))
    -cos(5)/2 + cos(1)/2
    """

    def f(rv):
        if not (rv.is_Mul or (rv.is_Pow and rv.base.func in (cos, sin) and (rv.exp.is_integer or rv.base.is_positive))):
            return rv
        if first:
            n, d = [expand_mul(i) for i in rv.as_numer_denom()]
            newn = TR8(n, first=False)
            newd = TR8(d, first=False)
            if newn != n or newd != d:
                rv = gcd_terms(newn / newd)
                if rv.is_Mul and rv.args[0].is_Rational and (len(rv.args) == 2) and rv.args[1].is_Add:
                    rv = Mul(*rv.as_coeff_Mul())
            return rv
        args = {cos: [], sin: [], None: []}
        for a in ordered(Mul.make_args(rv)):
            if a.func in (cos, sin):
                args[type(a)].append(a.args[0])
            elif a.is_Pow and a.exp.is_Integer and (a.exp > 0) and (a.base.func in (cos, sin)):
                args[type(a.base)].extend([a.base.args[0]] * a.exp)
            else:
                args[None].append(a)
        c = args[cos]
        s = args[sin]
        if not (c and s or len(c) > 1 or len(s) > 1):
            return rv
        args = args[None]
        n = min(len(c), len(s))
        for i in range(n):
            a1 = s.pop()
            a2 = c.pop()
            args.append((sin(a1 + a2) + sin(a1 - a2)) / 2)
        while len(c) > 1:
            a1 = c.pop()
            a2 = c.pop()
            args.append((cos(a1 + a2) + cos(a1 - a2)) / 2)
        if c:
            args.append(cos(c.pop()))
        while len(s) > 1:
            a1 = s.pop()
            a2 = s.pop()
            args.append((-cos(a1 + a2) + cos(a1 - a2)) / 2)
        if s:
            args.append(sin(s.pop()))
        return TR8(expand_mul(Mul(*args)))
    return bottom_up(rv, f)