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
def TR2i(rv, half=False):
    """Converts ratios involving sin and cos as follows::
        sin(x)/cos(x) -> tan(x)
        sin(x)/(cos(x) + 1) -> tan(x/2) if half=True

    Examples
    ========

    >>> from sympy.simplify.fu import TR2i
    >>> from sympy.abc import x, a
    >>> from sympy import sin, cos
    >>> TR2i(sin(x)/cos(x))
    tan(x)

    Powers of the numerator and denominator are also recognized

    >>> TR2i(sin(x)**2/(cos(x) + 1)**2, half=True)
    tan(x/2)**2

    The transformation does not take place unless assumptions allow
    (i.e. the base must be positive or the exponent must be an integer
    for both numerator and denominator)

    >>> TR2i(sin(x)**a/(cos(x) + 1)**a)
    sin(x)**a/(cos(x) + 1)**a

    """

    def f(rv):
        if not rv.is_Mul:
            return rv
        n, d = rv.as_numer_denom()
        if n.is_Atom or d.is_Atom:
            return rv

        def ok(k, e):
            return (e.is_integer or k.is_positive) and (k.func in (sin, cos) or (half and k.is_Add and (len(k.args) >= 2) and any((any((isinstance(ai, cos) or (ai.is_Pow and ai.base is cos) for ai in Mul.make_args(a))) for a in k.args))))
        n = n.as_powers_dict()
        ndone = [(k, n.pop(k)) for k in list(n.keys()) if not ok(k, n[k])]
        if not n:
            return rv
        d = d.as_powers_dict()
        ddone = [(k, d.pop(k)) for k in list(d.keys()) if not ok(k, d[k])]
        if not d:
            return rv

        def factorize(d, ddone):
            newk = []
            for k in d:
                if k.is_Add and len(k.args) > 1:
                    knew = factor(k) if half else factor_terms(k)
                    if knew != k:
                        newk.append((k, knew))
            if newk:
                for i, (k, knew) in enumerate(newk):
                    del d[k]
                    newk[i] = knew
                newk = Mul(*newk).as_powers_dict()
                for k in newk:
                    v = d[k] + newk[k]
                    if ok(k, v):
                        d[k] = v
                    else:
                        ddone.append((k, v))
                del newk
        factorize(n, ndone)
        factorize(d, ddone)
        t = []
        for k in n:
            if isinstance(k, sin):
                a = cos(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0]) ** n[k])
                    n[k] = d[a] = None
                elif half:
                    a1 = 1 + a
                    if a1 in d and d[a1] == n[k]:
                        t.append(tan(k.args[0] / 2) ** n[k])
                        n[k] = d[a1] = None
            elif isinstance(k, cos):
                a = sin(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0]) ** (-n[k]))
                    n[k] = d[a] = None
            elif half and k.is_Add and (k.args[0] is S.One) and isinstance(k.args[1], cos):
                a = sin(k.args[1].args[0], evaluate=False)
                if a in d and d[a] == n[k] and (d[a].is_integer or a.is_positive):
                    t.append(tan(a.args[0] / 2) ** (-n[k]))
                    n[k] = d[a] = None
        if t:
            rv = Mul(*t + [b ** e for b, e in n.items() if e]) / Mul(*[b ** e for b, e in d.items() if e])
            rv *= Mul(*[b ** e for b, e in ndone]) / Mul(*[b ** e for b, e in ddone])
        return rv
    return bottom_up(rv, f)