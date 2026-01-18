from math import prod
from collections import defaultdict
from typing import Tuple as tTuple
from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, Function, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import (binomial,
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib
class harmonic(Function):
    """
    Harmonic numbers

    The nth harmonic number is given by `\\operatorname{H}_{n} =
    1 + \\frac{1}{2} + \\frac{1}{3} + \\ldots + \\frac{1}{n}`.

    More generally:

    .. math:: \\operatorname{H}_{n,m} = \\sum_{k=1}^{n} \\frac{1}{k^m}

    As `n \\rightarrow \\infty`, `\\operatorname{H}_{n,m} \\rightarrow \\zeta(m)`,
    the Riemann zeta function.

    * ``harmonic(n)`` gives the nth harmonic number, `\\operatorname{H}_n`

    * ``harmonic(n, m)`` gives the nth generalized harmonic number
      of order `m`, `\\operatorname{H}_{n,m}`, where
      ``harmonic(n) == harmonic(n, 1)``

    This function can be extended to complex `n` and `m` where `n` is not a
    negative integer or `m` is a nonpositive integer as

    .. math:: \\operatorname{H}_{n,m} = \\begin{cases} \\zeta(m) - \\zeta(m, n+1)
            & m \\ne 1 \\\\ \\psi(n+1) + \\gamma & m = 1 \\end{cases}

    Examples
    ========

    >>> from sympy import harmonic, oo

    >>> [harmonic(n) for n in range(6)]
    [0, 1, 3/2, 11/6, 25/12, 137/60]
    >>> [harmonic(n, 2) for n in range(6)]
    [0, 1, 5/4, 49/36, 205/144, 5269/3600]
    >>> harmonic(oo, 2)
    pi**2/6

    >>> from sympy import Symbol, Sum
    >>> n = Symbol("n")

    >>> harmonic(n).rewrite(Sum)
    Sum(1/_k, (_k, 1, n))

    We can evaluate harmonic numbers for all integral and positive
    rational arguments:

    >>> from sympy import S, expand_func, simplify
    >>> harmonic(8)
    761/280
    >>> harmonic(11)
    83711/27720

    >>> H = harmonic(1/S(3))
    >>> H
    harmonic(1/3)
    >>> He = expand_func(H)
    >>> He
    -log(6) - sqrt(3)*pi/6 + 2*Sum(log(sin(_k*pi/3))*cos(2*_k*pi/3), (_k, 1, 1))
                           + 3*Sum(1/(3*_k + 1), (_k, 0, 0))
    >>> He.doit()
    -log(6) - sqrt(3)*pi/6 - log(sqrt(3)/2) + 3
    >>> H = harmonic(25/S(7))
    >>> He = simplify(expand_func(H).doit())
    >>> He
    log(sin(2*pi/7)**(2*cos(16*pi/7))/(14*sin(pi/7)**(2*cos(pi/7))*cos(pi/14)**(2*sin(pi/14)))) + pi*tan(pi/14)/2 + 30247/9900
    >>> He.n(40)
    1.983697455232980674869851942390639915940
    >>> harmonic(25/S(7)).n(40)
    1.983697455232980674869851942390639915940

    We can rewrite harmonic numbers in terms of polygamma functions:

    >>> from sympy import digamma, polygamma
    >>> m = Symbol("m", integer=True, positive=True)

    >>> harmonic(n).rewrite(digamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n).rewrite(polygamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n,3).rewrite(polygamma)
    polygamma(2, n + 1)/2 + zeta(3)

    >>> simplify(harmonic(n,m).rewrite(polygamma))
    Piecewise((polygamma(0, n + 1) + EulerGamma, Eq(m, 1)),
    (-(-1)**m*polygamma(m - 1, n + 1)/factorial(m - 1) + zeta(m), True))

    Integer offsets in the argument can be pulled out:

    >>> from sympy import expand_func

    >>> expand_func(harmonic(n+4))
    harmonic(n) + 1/(n + 4) + 1/(n + 3) + 1/(n + 2) + 1/(n + 1)

    >>> expand_func(harmonic(n-4))
    harmonic(n) - 1/(n - 1) - 1/(n - 2) - 1/(n - 3) - 1/n

    Some limits can be computed as well:

    >>> from sympy import limit, oo

    >>> limit(harmonic(n), n, oo)
    oo

    >>> limit(harmonic(n, 2), n, oo)
    pi**2/6

    >>> limit(harmonic(n, 3), n, oo)
    zeta(3)

    For `m > 1`, `H_{n,m}` tends to `\\zeta(m)` in the limit of infinite `n`:

    >>> m = Symbol("m", positive=True)
    >>> limit(harmonic(n, m+1), n, oo)
    zeta(m + 1)

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Harmonic_number
    .. [2] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber/
    .. [3] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber2/

    """

    @classmethod
    def eval(cls, n, m=None):
        from sympy.functions.special.zeta_functions import zeta
        if m is S.One:
            return cls(n)
        if m is None:
            m = S.One
        if n.is_zero:
            return S.Zero
        elif m.is_zero:
            return n
        elif n is S.Infinity:
            if m.is_negative:
                return S.NaN
            elif is_le(m, S.One):
                return S.Infinity
            elif is_gt(m, S.One):
                return zeta(m)
        elif m.is_Integer and m.is_nonpositive:
            return (bernoulli(1 - m, n + 1) - bernoulli(1 - m)) / (1 - m)
        elif n.is_Integer:
            if n.is_negative and (m.is_integer is False or m.is_nonpositive is False):
                return S.ComplexInfinity if m is S.One else S.NaN
            if n.is_nonnegative:
                return Add(*(k ** (-m) for k in range(1, int(n) + 1)))

    def _eval_rewrite_as_polygamma(self, n, m=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import gamma, polygamma
        if m.is_integer and m.is_positive:
            return Piecewise((polygamma(0, n + 1) + S.EulerGamma, Eq(m, 1)), (S.NegativeOne ** m * (polygamma(m - 1, 1) - polygamma(m - 1, n + 1)) / gamma(m), True))

    def _eval_rewrite_as_digamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_trigamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_Sum(self, n, m=None, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy('k', integer=True)
        if m is None:
            m = S.One
        return Sum(k ** (-m), (k, 1, n))

    def _eval_rewrite_as_zeta(self, n, m=S.One, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma(n + 1) + S.EulerGamma, Eq(m, 1)), (zeta(m) - zeta(m, n + 1), True))

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        n = self.args[0]
        m = self.args[1] if len(self.args) == 2 else 1
        if m == S.One:
            if n.is_Add:
                off = n.args[0]
                nnew = n - off
                if off.is_Integer and off.is_positive:
                    result = [S.One / (nnew + i) for i in range(off, 0, -1)] + [harmonic(nnew)]
                    return Add(*result)
                elif off.is_Integer and off.is_negative:
                    result = [-S.One / (nnew + i) for i in range(0, off, -1)] + [harmonic(nnew)]
                    return Add(*result)
            if n.is_Rational:
                p, q = n.as_numer_denom()
                u = p // q
                p = p - u * q
                if u.is_nonnegative and p.is_positive and q.is_positive and (p < q):
                    from sympy.functions.elementary.exponential import log
                    from sympy.functions.elementary.integers import floor
                    from sympy.functions.elementary.trigonometric import sin, cos, cot
                    k = Dummy('k')
                    t1 = q * Sum(1 / (q * k + p), (k, 0, u))
                    t2 = 2 * Sum(cos(2 * pi * p * k / S(q)) * log(sin(pi * k / S(q))), (k, 1, floor((q - 1) / S(2))))
                    t3 = pi / 2 * cot(pi * p / q) + log(2 * q)
                    return t1 + t2 - t3
        return self

    def _eval_rewrite_as_tractable(self, n, m=1, limitvar=None, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import polygamma
        pg = self.rewrite(polygamma)
        if not isinstance(pg, harmonic):
            return pg.rewrite('tractable', deep=True)
        arg = m - S.One
        if arg.is_nonzero:
            return (zeta(m) - zeta(m, n + 1)).rewrite('tractable', deep=True)

    def _eval_evalf(self, prec):
        if not all((x.is_number for x in self.args)):
            return
        n = self.args[0]._to_mpmath(prec)
        m = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        if mp.isint(n) and n < 0:
            return S.NaN
        with workprec(prec):
            if m == 1:
                res = mp.harmonic(n)
            else:
                res = mp.zeta(m) - mp.zeta(m, n + 1)
        return Expr._from_mpmath(res, prec)

    def fdiff(self, argindex=1):
        from sympy.functions.special.zeta_functions import zeta
        if len(self.args) == 2:
            n, m = self.args
        else:
            n, m = self.args + (1,)
        if argindex == 1:
            return m * zeta(m + 1, n + 1)
        else:
            raise ArgumentIndexError