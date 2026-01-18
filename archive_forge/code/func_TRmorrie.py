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
def TRmorrie(rv):
    """Returns cos(x)*cos(2*x)*...*cos(2**(k-1)*x) -> sin(2**k*x)/(2**k*sin(x))

    Examples
    ========

    >>> from sympy.simplify.fu import TRmorrie, TR8, TR3
    >>> from sympy.abc import x
    >>> from sympy import Mul, cos, pi
    >>> TRmorrie(cos(x)*cos(2*x))
    sin(4*x)/(4*sin(x))
    >>> TRmorrie(7*Mul(*[cos(x) for x in range(10)]))
    7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))

    Sometimes autosimplification will cause a power to be
    not recognized. e.g. in the following, cos(4*pi/7) automatically
    simplifies to -cos(3*pi/7) so only 2 of the 3 terms are
    recognized:

    >>> TRmorrie(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7))
    -sin(3*pi/7)*cos(3*pi/7)/(4*sin(pi/7))

    A touch by TR8 resolves the expression to a Rational

    >>> TR8(_)
    -1/8

    In this case, if eq is unsimplified, the answer is obtained
    directly:

    >>> eq = cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9)
    >>> TRmorrie(eq)
    1/16

    But if angles are made canonical with TR3 then the answer
    is not simplified without further work:

    >>> TR3(eq)
    sin(pi/18)*cos(pi/9)*cos(2*pi/9)/2
    >>> TRmorrie(_)
    sin(pi/18)*sin(4*pi/9)/(8*sin(pi/9))
    >>> TR8(_)
    cos(7*pi/18)/(16*sin(pi/9))
    >>> TR3(_)
    1/16

    The original expression would have resolve to 1/16 directly with TR8,
    however:

    >>> TR8(eq)
    1/16

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morrie%27s_law

    """

    def f(rv, first=True):
        if not rv.is_Mul:
            return rv
        if first:
            n, d = rv.as_numer_denom()
            return f(n, 0) / f(d, 0)
        args = defaultdict(list)
        coss = {}
        other = []
        for c in rv.args:
            b, e = c.as_base_exp()
            if e.is_Integer and isinstance(b, cos):
                co, a = b.args[0].as_coeff_Mul()
                args[a].append(co)
                coss[b] = e
            else:
                other.append(c)
        new = []
        for a in args:
            c = args[a]
            c.sort()
            while c:
                k = 0
                cc = ci = c[0]
                while cc in c:
                    k += 1
                    cc *= 2
                if k > 1:
                    newarg = sin(2 ** k * ci * a) / 2 ** k / sin(ci * a)
                    take = None
                    ccs = []
                    for i in range(k):
                        cc /= 2
                        key = cos(a * cc, evaluate=False)
                        ccs.append(cc)
                        take = min(coss[key], take or coss[key])
                    for i in range(k):
                        cc = ccs.pop()
                        key = cos(a * cc, evaluate=False)
                        coss[key] -= take
                        if not coss[key]:
                            c.remove(cc)
                    new.append(newarg ** take)
                else:
                    b = cos(c.pop(0) * a)
                    other.append(b ** coss[b])
        if new:
            rv = Mul(*new + other + [cos(k * a, evaluate=False) for a in args for k in args[a]])
        return rv
    return bottom_up(rv, f)