from collections import defaultdict
from functools import reduce
import random
import math
from sympy.core import sympify
from sympy.core.containers import Dict
from sympy.core.evalf import bitcount
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm, Rational, Integer
from sympy.core.power import integer_nthroot, Pow, integer_log
from sympy.core.singleton import S
from sympy.external.gmpy import SYMPY_INTS
from .primetest import isprime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor
def _check_termination(factors, n, limitp1, use_trial, use_rho, use_pm1, verbose):
    """
    Helper function for integer factorization. Checks if ``n``
    is a prime or a perfect power, and in those cases updates
    the factorization and raises ``StopIteration``.
    """
    if verbose:
        print('Check for termination')
    p = perfect_power(n, factor=False)
    if p is not False:
        base, exp = p
        if limitp1:
            limit = limitp1 - 1
        else:
            limit = limitp1
        facs = factorint(base, limit, use_trial, use_rho, use_pm1, verbose=False)
        for b, e in facs.items():
            if verbose:
                print(factor_msg % (b, e))
            factors[b] = exp * e
        raise StopIteration
    if isprime(n):
        factors[int(n)] = 1
        raise StopIteration
    if n == 1:
        raise StopIteration