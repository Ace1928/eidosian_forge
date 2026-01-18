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