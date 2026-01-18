import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv, ndtr, ndtri
from scipy.stats._qmc import primes_from_2_to
def _factorize_int(n):
    """Return a sorted list of the unique prime factors of a positive integer.
    """
    factors = set()
    for p in primes_from_2_to(int(np.sqrt(n)) + 1):
        while not n % p:
            factors.add(p)
            n //= p
        if n == 1:
            break
    if n != 1:
        factors.add(n)
    return sorted(factors)