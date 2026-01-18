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
class bernoulli(Function):
    """
    Bernoulli numbers / Bernoulli polynomials / Bernoulli function

    The Bernoulli numbers are a sequence of rational numbers
    defined by `B_0 = 1` and the recursive relation (`n > 0`):

    .. math :: n+1 = \\sum_{k=0}^n \\binom{n+1}{k} B_k

    They are also commonly defined by their exponential generating
    function, which is `\\frac{x}{1 - e^{-x}}`. For odd indices > 1,
    the Bernoulli numbers are zero.

    The Bernoulli polynomials satisfy the analogous formula:

    .. math :: B_n(x) = \\sum_{k=0}^n (-1)^k \\binom{n}{k} B_k x^{n-k}

    Bernoulli numbers and Bernoulli polynomials are related as
    `B_n(1) = B_n`.

    The generalized Bernoulli function `\\operatorname{B}(s, a)`
    is defined for any complex `s` and `a`, except where `a` is a
    nonpositive integer and `s` is not a nonnegative integer. It is
    an entire function of `s` for fixed `a`, related to the Hurwitz
    zeta function by

    .. math:: \\operatorname{B}(s, a) = \\begin{cases}
              -s \\zeta(1-s, a) & s \\ne 0 \\\\ 1 & s = 0 \\end{cases}

    When `s` is a nonnegative integer this function reduces to the
    Bernoulli polynomials: `\\operatorname{B}(n, x) = B_n(x)`. When
    `a` is omitted it is assumed to be 1, yielding the (ordinary)
    Bernoulli function which interpolates the Bernoulli numbers and is
    related to the Riemann zeta function.

    We compute Bernoulli numbers using Ramanujan's formula:

    .. math :: B_n = \\frac{A(n) - S(n)}{\\binom{n+3}{n}}

    where:

    .. math :: A(n) = \\begin{cases} \\frac{n+3}{3} &
        n \\equiv 0\\ \\text{or}\\ 2 \\pmod{6} \\\\
        -\\frac{n+3}{6} & n \\equiv 4 \\pmod{6} \\end{cases}

    and:

    .. math :: S(n) = \\sum_{k=1}^{[n/6]} \\binom{n+3}{n-6k} B_{n-6k}

    This formula is similar to the sum given in the definition, but
    cuts `\\frac{2}{3}` of the terms. For Bernoulli polynomials, we use
    Appell sequences.

    For `n` a nonnegative integer and `s`, `a`, `x` arbitrary complex numbers,

    * ``bernoulli(n)`` gives the nth Bernoulli number, `B_n`
    * ``bernoulli(s)`` gives the Bernoulli function `\\operatorname{B}(s)`
    * ``bernoulli(n, x)`` gives the nth Bernoulli polynomial in `x`, `B_n(x)`
    * ``bernoulli(s, a)`` gives the generalized Bernoulli function
      `\\operatorname{B}(s, a)`

    .. versionchanged:: 1.12
        ``bernoulli(1)`` gives `+\\frac{1}{2}` instead of `-\\frac{1}{2}`.
        This choice of value confers several theoretical advantages [5]_,
        including the extension to complex parameters described above
        which this function now implements. The previous behavior, defined
        only for nonnegative integers `n`, can be obtained with
        ``(-1)**n*bernoulli(n)``.

    Examples
    ========

    >>> from sympy import bernoulli
    >>> from sympy.abc import x
    >>> [bernoulli(n) for n in range(11)]
    [1, 1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> bernoulli(1000001)
    0
    >>> bernoulli(3, x)
    x**3 - 3*x**2/2 + x/2

    See Also
    ========

    andre, bell, catalan, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.bernoulli_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_number
    .. [2] https://en.wikipedia.org/wiki/Bernoulli_polynomial
    .. [3] https://mathworld.wolfram.com/BernoulliNumber.html
    .. [4] https://mathworld.wolfram.com/BernoulliPolynomial.html
    .. [5] Peter Luschny, "The Bernoulli Manifesto",
           https://luschny.de/math/zeta/The-Bernoulli-Manifesto.html
    .. [6] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """
    args: tTuple[Integer]

    @staticmethod
    def _calc_bernoulli(n):
        s = 0
        a = int(binomial(n + 3, n - 6))
        for j in range(1, n // 6 + 1):
            s += a * bernoulli(n - 6 * j)
            a *= _product(n - 6 - 6 * j + 1, n - 6 * j)
            a //= _product(6 * j + 4, 6 * j + 9)
        if n % 6 == 4:
            s = -Rational(n + 3, 6) - s
        else:
            s = Rational(n + 3, 3) - s
        return s / binomial(n + 3, n)
    _cache = {0: S.One, 2: Rational(1, 6), 4: Rational(-1, 30)}
    _highest = {0: 0, 2: 2, 4: 4}

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_zero:
            return S.One
        elif n.is_integer is False or n.is_nonnegative is False:
            if x is not None and x.is_Integer and x.is_nonpositive:
                return S.NaN
            return
        elif x is None:
            if n is S.One:
                return S.Half
            elif n.is_odd and (n - 1).is_positive:
                return S.Zero
            elif n.is_Number:
                n = int(n)
                if n > 500:
                    p, q = mp.bernfrac(n)
                    return Rational(int(p), int(q))
                case = n % 6
                highest_cached = cls._highest[case]
                if n <= highest_cached:
                    return cls._cache[n]
                for i in range(highest_cached + 6, n + 6, 6):
                    b = cls._calc_bernoulli(i)
                    cls._cache[i] = b
                    cls._highest[case] = i
                return b
        elif n.is_Number:
            return bernoulli_poly(n, x)

    def _eval_rewrite_as_zeta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        return Piecewise((1, Eq(n, 0)), (-n * zeta(1 - n, x), True))

    def _eval_evalf(self, prec):
        if not all((x.is_number for x in self.args)):
            return
        n = self.args[0]._to_mpmath(prec)
        x = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        with workprec(prec):
            if n == 0:
                res = mp.mpf(1)
            elif n == 1:
                res = x - mp.mpf(0.5)
            elif mp.isint(n) and n >= 0:
                res = mp.bernoulli(n) if x == 1 else mp.bernpoly(n, x)
            else:
                res = -n * mp.zeta(1 - n, x)
        return Expr._from_mpmath(res, prec)