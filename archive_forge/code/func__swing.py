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