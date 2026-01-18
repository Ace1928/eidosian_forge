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
def _factorint_small(factors, n, limit, fail_max):
    """
    Return the value of n and either a 0 (indicating that factorization up
    to the limit was complete) or else the next near-prime that would have
    been tested.

    Factoring stops if there are fail_max unsuccessful tests in a row.

    If factors of n were found they will be in the factors dictionary as
    {factor: multiplicity} and the returned value of n will have had those
    factors removed. The factors dictionary is modified in-place.

    """

    def done(n, d):
        """return n, d if the sqrt(n) was not reached yet, else
           n, 0 indicating that factoring is done.
        """
        if d * d <= n:
            return (n, d)
        return (n, 0)
    d = 2
    m = trailing(n)
    if m:
        factors[d] = m
        n >>= m
    d = 3
    if limit < d:
        if n > 1:
            factors[n] = 1
        return done(n, d)
    m = 0
    while n % d == 0:
        n //= d
        m += 1
        if m == 20:
            mm = multiplicity(d, n)
            m += mm
            n //= d ** mm
            break
    if m:
        factors[d] = m
    if limit * limit > n:
        maxx = 0
    else:
        maxx = limit * limit
    dd = maxx or n
    d = 5
    fails = 0
    while fails < fail_max:
        if d * d > dd:
            break
        m = 0
        while n % d == 0:
            n //= d
            m += 1
            if m == 20:
                mm = multiplicity(d, n)
                m += mm
                n //= d ** mm
                break
        if m:
            factors[d] = m
            dd = maxx or n
            fails = 0
        else:
            fails += 1
        d += 2
        if d * d > dd:
            break
        m = 0
        while n % d == 0:
            n //= d
            m += 1
            if m == 20:
                mm = multiplicity(d, n)
                m += mm
                n //= d ** mm
                break
        if m:
            factors[d] = m
            dd = maxx or n
            fails = 0
        else:
            fails += 1
        d += 4
    return done(n, d)