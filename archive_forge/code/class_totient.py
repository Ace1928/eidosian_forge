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
class totient(Function):
    """
    Calculate the Euler totient function phi(n)

    ``totient(n)`` or `\\phi(n)` is the number of positive integers `\\leq` n
    that are relatively prime to n.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.ntheory import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html

    """

    @classmethod
    def eval(cls, n):
        if n.is_Integer:
            if n < 1:
                raise ValueError('n must be a positive integer')
            factors = factorint(n)
            return cls._from_factors(factors)
        elif not isinstance(n, Expr) or n.is_integer is False or n.is_positive is False:
            raise ValueError('n must be a positive integer')

    def _eval_is_integer(self):
        return fuzzy_and([self.args[0].is_integer, self.args[0].is_positive])

    @classmethod
    def _from_distinct_primes(self, *args):
        """Subroutine to compute totient from the list of assumed
        distinct primes

        Examples
        ========

        >>> from sympy.ntheory.factor_ import totient
        >>> totient._from_distinct_primes(5, 7)
        24
        """
        return reduce(lambda i, j: i * (j - 1), args, 1)

    @classmethod
    def _from_factors(self, factors):
        """Subroutine to compute totient from already-computed factors

        Examples
        ========

        >>> from sympy.ntheory.factor_ import totient
        >>> totient._from_factors({5: 2})
        20
        """
        t = 1
        for p, k in factors.items():
            t *= (p - 1) * p ** (k - 1)
        return t