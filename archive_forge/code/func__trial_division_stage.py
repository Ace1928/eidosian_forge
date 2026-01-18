from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _trial_division_stage(N, M, factor_base, sieve_array, sieve_poly, partial_relations, ERROR_TERM):
    """Trial division stage. Here we trial divide the values generetated
    by sieve_poly in the sieve interval and if it is a smooth number then
    it is stored in `smooth_relations`. Moreover, if we find two partial relations
    with same large prime then they are combined to form a smooth relation.
    First we iterate over sieve array and look for values which are greater
    than accumulated_val, as these values have a high chance of being smooth
    number. Then using these values we find smooth relations.
    In general, let ``t**2 = u*p modN`` and ``r**2 = v*p modN`` be two partial relations
    with the same large prime p. Then they can be combined ``(t*r/p)**2 = u*v modN``
    to form a smooth relation.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    sieve_array : stores log_p values
    sieve_poly : polynomial from which we find smooth relations
    partial_relations : stores partial relations with one large prime
    ERROR_TERM : error term for accumulated_val
    """
    sqrt_n = sqrt(float(N))
    accumulated_val = log(M * sqrt_n) * 2 ** 10 - ERROR_TERM
    smooth_relations = []
    proper_factor = set()
    partial_relation_upper_bound = 128 * factor_base[-1].prime
    for idx, val in enumerate(sieve_array):
        if val < accumulated_val:
            continue
        x = idx - M
        v = sieve_poly.eval(x)
        vec, is_smooth = _check_smoothness(v, factor_base)
        if is_smooth is None:
            continue
        u = sieve_poly.a * x + sieve_poly.b
        if is_smooth is False:
            large_prime = vec
            if large_prime > partial_relation_upper_bound:
                continue
            if large_prime not in partial_relations:
                partial_relations[large_prime] = (u, v)
                continue
            else:
                u_prev, v_prev = partial_relations[large_prime]
                partial_relations.pop(large_prime)
                try:
                    large_prime_inv = mod_inverse(large_prime, N)
                except ValueError:
                    proper_factor.add(large_prime)
                    continue
                u = u * u_prev * large_prime_inv
                v = v * v_prev // (large_prime * large_prime)
                vec, is_smooth = _check_smoothness(v, factor_base)
        smooth_relations.append((u, v, vec))
    return (smooth_relations, proper_factor)