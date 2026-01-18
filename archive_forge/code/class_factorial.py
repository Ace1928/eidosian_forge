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
class factorial(CombinatorialFunction):
    """Implementation of factorial function over nonnegative integers.
       By convention (consistent with the gamma function and the binomial
       coefficients), factorial of a negative integer is complex infinity.

       The factorial is very important in combinatorics where it gives
       the number of ways in which `n` objects can be permuted. It also
       arises in calculus, probability, number theory, etc.

       There is strict relation of factorial with gamma function. In
       fact `n! = gamma(n+1)` for nonnegative integers. Rewrite of this
       kind is very useful in case of combinatorial simplification.

       Computation of the factorial is done using two algorithms. For
       small arguments a precomputed look up table is used. However for bigger
       input algorithm Prime-Swing is used. It is the fastest algorithm
       known and computes `n!` via prime factorization of special class
       of numbers, called here the 'Swing Numbers'.

       Examples
       ========

       >>> from sympy import Symbol, factorial, S
       >>> n = Symbol('n', integer=True)

       >>> factorial(0)
       1

       >>> factorial(7)
       5040

       >>> factorial(-2)
       zoo

       >>> factorial(n)
       factorial(n)

       >>> factorial(2*n)
       factorial(2*n)

       >>> factorial(S(1)/2)
       factorial(1/2)

       See Also
       ========

       factorial2, RisingFactorial, FallingFactorial
    """

    def fdiff(self, argindex=1):
        from sympy.functions.special.gamma_functions import gamma, polygamma
        if argindex == 1:
            return gamma(self.args[0] + 1) * polygamma(0, self.args[0] + 1)
        else:
            raise ArgumentIndexError(self, argindex)
    _small_swing = [1, 1, 1, 3, 3, 15, 5, 35, 35, 315, 63, 693, 231, 3003, 429, 6435, 6435, 109395, 12155, 230945, 46189, 969969, 88179, 2028117, 676039, 16900975, 1300075, 35102025, 5014575, 145422675, 9694845, 300540195, 300540195]
    _small_factorials: list[int] = []

    @classmethod
    def _swing(cls, n):
        if n < 33:
            return cls._small_swing[n]
        else:
            N, primes = (int(_sqrt(n)), [])
            for prime in sieve.primerange(3, N + 1):
                p, q = (1, n)
                while True:
                    q //= prime
                    if q > 0:
                        if q & 1 == 1:
                            p *= prime
                    else:
                        break
                if p > 1:
                    primes.append(p)
            for prime in sieve.primerange(N + 1, n // 3 + 1):
                if n // prime & 1 == 1:
                    primes.append(prime)
            L_product = prod(sieve.primerange(n // 2 + 1, n + 1))
            R_product = prod(primes)
            return L_product * R_product

    @classmethod
    def _recursive(cls, n):
        if n < 2:
            return 1
        else:
            return cls._recursive(n // 2) ** 2 * cls._swing(n)

    @classmethod
    def eval(cls, n):
        n = sympify(n)
        if n.is_Number:
            if n.is_zero:
                return S.One
            elif n is S.Infinity:
                return S.Infinity
            elif n.is_Integer:
                if n.is_negative:
                    return S.ComplexInfinity
                else:
                    n = n.p
                    if n < 20:
                        if not cls._small_factorials:
                            result = 1
                            for i in range(1, 20):
                                result *= i
                                cls._small_factorials.append(result)
                        result = cls._small_factorials[n - 1]
                    elif HAS_GMPY:
                        result = gmpy.fac(n)
                    else:
                        bits = bin(n).count('1')
                        result = cls._recursive(n) * 2 ** (n - bits)
                    return Integer(result)

    def _facmod(self, n, q):
        res, N = (1, int(_sqrt(n)))
        pw = [1] * N
        m = 2
        for prime in sieve.primerange(2, n + 1):
            if m > 1:
                m, y = (0, n // prime)
                while y:
                    m += y
                    y //= prime
            if m < N:
                pw[m] = pw[m] * prime % q
            else:
                res = res * pow(prime, m, q) % q
        for ex, bs in enumerate(pw):
            if ex == 0 or bs == 1:
                continue
            if bs == 0:
                return 0
            res = res * pow(bs, ex, q) % q
        return res

    def _eval_Mod(self, q):
        n = self.args[0]
        if n.is_integer and n.is_nonnegative and q.is_integer:
            aq = abs(q)
            d = aq - n
            if d.is_nonpositive:
                return S.Zero
            else:
                isprime = aq.is_prime
                if d == 1:
                    if isprime:
                        return -1 % q
                    elif isprime is False and (aq - 6).is_nonnegative:
                        return S.Zero
                elif n.is_Integer and q.is_Integer:
                    n, d, aq = map(int, (n, d, aq))
                    if isprime and d - 1 < n:
                        fc = self._facmod(d - 1, aq)
                        fc = pow(fc, aq - 2, aq)
                        if d % 2:
                            fc = -fc
                    else:
                        fc = self._facmod(n, aq)
                    return fc % q

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return gamma(n + 1)

    def _eval_rewrite_as_Product(self, n, **kwargs):
        from sympy.concrete.products import Product
        if n.is_nonnegative and n.is_integer:
            i = Dummy('i', integer=True)
            return Product(i, (i, 1, n))

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_even(self):
        x = self.args[0]
        if x.is_integer and x.is_nonnegative:
            return (x - 2).is_nonnegative

    def _eval_is_composite(self):
        x = self.args[0]
        if x.is_integer and x.is_nonnegative:
            return (x - 3).is_nonnegative

    def _eval_is_real(self):
        x = self.args[0]
        if x.is_nonnegative or x.is_noninteger:
            return True

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x)
        arg0 = arg.subs(x, 0)
        if arg0.is_zero:
            return S.One
        elif not arg0.is_infinite:
            return self.func(arg)
        raise PoleError('Cannot expand %s around 0' % self)