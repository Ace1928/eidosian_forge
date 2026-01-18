from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
class FactorBaseElem:
    """This class stores an element of the `factor_base`.
    """

    def __init__(self, prime, tmem_p, log_p):
        """
        Initialization of factor_base_elem.

        Parameters
        ==========

        prime : prime number of the factor_base
        tmem_p : Integer square root of x**2 = n mod prime
        log_p : Compute Natural Logarithm of the prime
        """
        self.prime = prime
        self.tmem_p = tmem_p
        self.log_p = log_p
        self.soln1 = None
        self.soln2 = None
        self.a_inv = None
        self.b_ainv = None