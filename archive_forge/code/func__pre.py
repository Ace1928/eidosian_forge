from mpmath.libmp import (fzero, from_int, from_rational,
from sympy.core.numbers import igcd
from .residue_ntheory import (_sqrt_mod_prime_power,
import math
def _pre():
    maxn = 10 ** 5
    global _factor
    global _totient
    _factor = [0] * maxn
    _totient = [1] * maxn
    lim = int(maxn ** 0.5) + 5
    for i in range(2, lim):
        if _factor[i] == 0:
            for j in range(i * i, maxn, i):
                if _factor[j] == 0:
                    _factor[j] = i
    for i in range(2, maxn):
        if _factor[i] == 0:
            _factor[i] = i
            _totient[i] = i - 1
            continue
        x = _factor[i]
        y = i // x
        if y % x == 0:
            _totient[i] = _totient[y] * x
        else:
            _totient[i] = _totient[y] * (x - 1)