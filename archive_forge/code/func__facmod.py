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