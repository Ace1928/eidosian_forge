from __future__ import annotations
from functools import reduce
from sympy.core import S, sympify, Dummy, Mod
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
from sympy.core.numbers import Integer, pi, I
from sympy.core.relational import Eq
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.ntheory import sieve
from sympy.polys.polytools import Poly
from math import factorial as _factorial, prod, sqrt as _sqrt
class binomial(CombinatorialFunction):
    """Implementation of the binomial coefficient. It can be defined
    in two ways depending on its desired interpretation:

    .. math:: \\binom{n}{k} = \\frac{n!}{k!(n-k)!}\\ \\text{or}\\
                \\binom{n}{k} = \\frac{(n)_k}{k!}

    First, in a strict combinatorial sense it defines the
    number of ways we can choose `k` elements from a set of
    `n` elements. In this case both arguments are nonnegative
    integers and binomial is computed using an efficient
    algorithm based on prime factorization.

    The other definition is generalization for arbitrary `n`,
    however `k` must also be nonnegative. This case is very
    useful when evaluating summations.

    For the sake of convenience, for negative integer `k` this function
    will return zero no matter the other argument.

    To expand the binomial when `n` is a symbol, use either
    ``expand_func()`` or ``expand(func=True)``. The former will keep
    the polynomial in factored form while the latter will expand the
    polynomial itself. See examples for details.

    Examples
    ========

    >>> from sympy import Symbol, Rational, binomial, expand_func
    >>> n = Symbol('n', integer=True, positive=True)

    >>> binomial(15, 8)
    6435

    >>> binomial(n, -1)
    0

    Rows of Pascal's triangle can be generated with the binomial function:

    >>> for N in range(8):
    ...     print([binomial(N, i) for i in range(N + 1)])
    ...
    [1]
    [1, 1]
    [1, 2, 1]
    [1, 3, 3, 1]
    [1, 4, 6, 4, 1]
    [1, 5, 10, 10, 5, 1]
    [1, 6, 15, 20, 15, 6, 1]
    [1, 7, 21, 35, 35, 21, 7, 1]

    As can a given diagonal, e.g. the 4th diagonal:

    >>> N = -4
    >>> [binomial(N, i) for i in range(1 - N)]
    [1, -4, 10, -20, 35]

    >>> binomial(Rational(5, 4), 3)
    -5/128
    >>> binomial(Rational(-5, 4), 3)
    -195/128

    >>> binomial(n, 3)
    binomial(n, 3)

    >>> binomial(n, 3).expand(func=True)
    n**3/6 - n**2/2 + n/3

    >>> expand_func(binomial(n, 3))
    n*(n - 2)*(n - 1)/6

    References
    ==========

    .. [1] https://www.johndcook.com/blog/binomial_coefficients/

    """

    def fdiff(self, argindex=1):
        from sympy.functions.special.gamma_functions import polygamma
        if argindex == 1:
            n, k = self.args
            return binomial(n, k) * (polygamma(0, n + 1) - polygamma(0, n - k + 1))
        elif argindex == 2:
            n, k = self.args
            return binomial(n, k) * (polygamma(0, n - k + 1) - polygamma(0, k + 1))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def _eval(self, n, k):
        if k.is_Integer:
            if n.is_Integer and n >= 0:
                n, k = (int(n), int(k))
                if k > n:
                    return S.Zero
                elif k > n // 2:
                    k = n - k
                if HAS_GMPY:
                    return Integer(gmpy.bincoef(n, k))
                d, result = (n - k, 1)
                for i in range(1, k + 1):
                    d += 1
                    result = result * d // i
                return Integer(result)
            else:
                d, result = (n - k, 1)
                for i in range(1, k + 1):
                    d += 1
                    result *= d
                return result / _factorial(k)

    @classmethod
    def eval(cls, n, k):
        n, k = map(sympify, (n, k))
        d = n - k
        n_nonneg, n_isint = (n.is_nonnegative, n.is_integer)
        if k.is_zero or ((n_nonneg or n_isint is False) and d.is_zero):
            return S.One
        if (k - 1).is_zero or ((n_nonneg or n_isint is False) and (d - 1).is_zero):
            return n
        if k.is_integer:
            if k.is_negative or (n_nonneg and n_isint and d.is_negative):
                return S.Zero
            elif n.is_number:
                res = cls._eval(n, k)
                return res.expand(basic=True) if res else res
        elif n_nonneg is False and n_isint:
            return S.ComplexInfinity
        elif k.is_number:
            from sympy.functions.special.gamma_functions import gamma
            return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

    def _eval_Mod(self, q):
        n, k = self.args
        if any((x.is_integer is False for x in (n, k, q))):
            raise ValueError('Integers expected for binomial Mod')
        if all((x.is_Integer for x in (n, k, q))):
            n, k = map(int, (n, k))
            aq, res = (abs(q), 1)
            if k < 0:
                return S.Zero
            if n < 0:
                n = -n + k - 1
                res = -1 if k % 2 else 1
            if k > n:
                return S.Zero
            isprime = aq.is_prime
            aq = int(aq)
            if isprime:
                if aq < n:
                    N, K = (n, k)
                    while N or K:
                        res = res * binomial(N % aq, K % aq) % aq
                        N, K = (N // aq, K // aq)
                else:
                    d = n - k
                    if k > d:
                        k, d = (d, k)
                    kf = 1
                    for i in range(2, k + 1):
                        kf = kf * i % aq
                    df = kf
                    for i in range(k + 1, d + 1):
                        df = df * i % aq
                    res *= df
                    for i in range(d + 1, n + 1):
                        res = res * i % aq
                    res *= pow(kf * df % aq, aq - 2, aq)
                    res %= aq
            else:
                M = int(_sqrt(n))
                for prime in sieve.primerange(2, n + 1):
                    if prime > n - k:
                        res = res * prime % aq
                    elif prime > n // 2:
                        continue
                    elif prime > M:
                        if n % prime < k % prime:
                            res = res * prime % aq
                    else:
                        N, K = (n, k)
                        exp = a = 0
                        while N > 0:
                            a = int(N % prime < K % prime + a)
                            N, K = (N // prime, K // prime)
                            exp += a
                        if exp > 0:
                            res *= pow(prime, exp, aq)
                            res %= aq
            return S(res % q)

    def _eval_expand_func(self, **hints):
        """
        Function to expand binomial(n, k) when m is positive integer
        Also,
        n is self.args[0] and k is self.args[1] while using binomial(n, k)
        """
        n = self.args[0]
        if n.is_Number:
            return binomial(*self.args)
        k = self.args[1]
        if (n - k).is_Integer:
            k = n - k
        if k.is_Integer:
            if k.is_zero:
                return S.One
            elif k.is_negative:
                return S.Zero
            else:
                n, result = (self.args[0], 1)
                for i in range(1, k + 1):
                    result *= n - k + i
                return result / _factorial(k)
        else:
            return binomial(*self.args)

    def _eval_rewrite_as_factorial(self, n, k, **kwargs):
        return factorial(n) / (factorial(k) * factorial(n - k))

    def _eval_rewrite_as_gamma(self, n, k, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

    def _eval_rewrite_as_tractable(self, n, k, limitvar=None, **kwargs):
        return self._eval_rewrite_as_gamma(n, k).rewrite('tractable')

    def _eval_rewrite_as_FallingFactorial(self, n, k, **kwargs):
        if k.is_integer:
            return ff(n, k) / factorial(k)

    def _eval_is_integer(self):
        n, k = self.args
        if n.is_integer and k.is_integer:
            return True
        elif k.is_integer is False:
            return False

    def _eval_is_nonnegative(self):
        n, k = self.args
        if n.is_integer and k.is_integer:
            if n.is_nonnegative or k.is_negative or k.is_even:
                return True
            elif k.is_even is False:
                return False

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.functions.special.gamma_functions import gamma
        return self.rewrite(gamma)._eval_as_leading_term(x, logx=logx, cdir=cdir)