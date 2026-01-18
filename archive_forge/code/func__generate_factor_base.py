from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _generate_factor_base(prime_bound, n):
    """Generate `factor_base` for Quadratic Sieve. The `factor_base`
    consists of all the points whose ``legendre_symbol(n, p) == 1``
    and ``p < num_primes``. Along with the prime `factor_base` also stores
    natural logarithm of prime and the residue n modulo p.
    It also returns the of primes numbers in the `factor_base` which are
    close to 1000 and 5000.

    Parameters
    ==========

    prime_bound : upper prime bound of the factor_base
    n : integer to be factored
    """
    from sympy.ntheory.generate import sieve
    factor_base = []
    idx_1000, idx_5000 = (None, None)
    for prime in sieve.primerange(1, prime_bound):
        if pow(n, (prime - 1) // 2, prime) == 1:
            if prime > 1000 and idx_1000 is None:
                idx_1000 = len(factor_base) - 1
            if prime > 5000 and idx_5000 is None:
                idx_5000 = len(factor_base) - 1
            residue = _sqrt_mod_prime_power(n, prime, 1)[0]
            log_p = round(log(prime) * 2 ** 10)
            factor_base.append(FactorBaseElem(prime, residue, log_p))
    return (idx_1000, idx_5000, factor_base)