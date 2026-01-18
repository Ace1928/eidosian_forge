from __future__ import annotations
from sympy.core.function import Function
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity, perfect_power
from sympy.utilities.misc import as_int
from sympy.core.random import _randint, randint
from itertools import cycle, product
def _polynomial_congruence_prime(coefficients, p):
    """A helper function used by polynomial_congruence.
    It returns the root of a polynomial modulo prime number
    by naive search from [0, p).

    Parameters
    ==========

    coefficients : list of integers
    p : prime number
    """
    roots = []
    rank = len(coefficients)
    for i in range(0, p):
        f_val = 0
        for coeff in range(0, rank - 1):
            f_val = (f_val + pow(i, int(rank - coeff - 1), p) * coefficients[coeff]) % p
        f_val = f_val + coefficients[-1]
        if f_val % p == 0:
            roots.append(i)
    return roots