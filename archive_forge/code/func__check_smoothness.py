from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _check_smoothness(num, factor_base):
    """Here we check that if `num` is a smooth number or not. If `a` is a smooth
    number then it returns a vector of prime exponents modulo 2. For example
    if a = 2 * 5**2 * 7**3 and the factor base contains {2, 3, 5, 7} then
    `a` is a smooth number and this function returns ([1, 0, 0, 1], True). If
    `a` is a partial relation which means that `a` a has one prime factor
    greater than the `factor_base` then it returns `(a, False)` which denotes `a`
    is a partial relation.

    Parameters
    ==========

    a : integer whose smootheness is to be checked
    factor_base : factor_base primes
    """
    vec = []
    if num < 0:
        vec.append(1)
        num *= -1
    else:
        vec.append(0)
    for factor in factor_base:
        if num % factor.prime != 0:
            vec.append(0)
            continue
        factor_exp = 0
        while num % factor.prime == 0:
            factor_exp += 1
            num //= factor.prime
        vec.append(factor_exp % 2)
    if num == 1:
        return (vec, True)
    if isprime(num):
        return (num, False)
    return (None, None)