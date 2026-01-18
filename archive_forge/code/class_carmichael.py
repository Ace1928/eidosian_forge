from math import prod
from collections import defaultdict
from typing import Tuple as tTuple
from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, Function, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import (binomial,
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib
class carmichael(Function):
    """
    Carmichael Numbers:

    Certain cryptographic algorithms make use of big prime numbers.
    However, checking whether a big number is prime is not so easy.
    Randomized prime number checking tests exist that offer a high degree of
    confidence of accurate determination at low cost, such as the Fermat test.

    Let 'a' be a random number between $2$ and $n - 1$, where $n$ is the
    number whose primality we are testing. Then, $n$ is probably prime if it
    satisfies the modular arithmetic congruence relation:

    .. math :: a^{n-1} = 1 \\pmod{n}

    (where mod refers to the modulo operation)

    If a number passes the Fermat test several times, then it is prime with a
    high probability.

    Unfortunately, certain composite numbers (non-primes) still pass the Fermat
    test with every number smaller than themselves.
    These numbers are called Carmichael numbers.

    A Carmichael number will pass a Fermat primality test to every base $b$
    relatively prime to the number, even though it is not actually prime.
    This makes tests based on Fermat's Little Theorem less effective than
    strong probable prime tests such as the Baillie-PSW primality test and
    the Miller-Rabin primality test.

    Examples
    ========

    >>> from sympy import carmichael
    >>> carmichael.find_first_n_carmichaels(5)
    [561, 1105, 1729, 2465, 2821]
    >>> carmichael.find_carmichael_numbers_in_range(0, 562)
    [561]
    >>> carmichael.find_carmichael_numbers_in_range(0,1000)
    [561]
    >>> carmichael.find_carmichael_numbers_in_range(0,2000)
    [561, 1105, 1729]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_number
    .. [2] https://en.wikipedia.org/wiki/Fermat_primality_test
    .. [3] https://www.jstor.org/stable/23248683?seq=1#metadata_info_tab_contents
    """

    @staticmethod
    def is_perfect_square(n):
        sympy_deprecation_warning('\nis_perfect_square is just a wrapper around sympy.ntheory.primetest.is_square\nso use that directly instead.\n        ', deprecated_since_version='1.11', active_deprecations_target='deprecated-carmichael-static-methods')
        return is_square(n)

    @staticmethod
    def divides(p, n):
        sympy_deprecation_warning('\n        divides can be replaced by directly testing n % p == 0.\n        ', deprecated_since_version='1.11', active_deprecations_target='deprecated-carmichael-static-methods')
        return n % p == 0

    @staticmethod
    def is_prime(n):
        sympy_deprecation_warning('\nis_prime is just a wrapper around sympy.ntheory.primetest.isprime so use that\ndirectly instead.\n        ', deprecated_since_version='1.11', active_deprecations_target='deprecated-carmichael-static-methods')
        return isprime(n)

    @staticmethod
    def is_carmichael(n):
        if n >= 0:
            if n == 1 or isprime(n) or n % 2 == 0:
                return False
            divisors = [1, n]
            divisors.extend([i for i in range(3, n // 2 + 1, 2) if n % i == 0])
            for i in divisors:
                if is_square(i) and i != 1:
                    return False
                if isprime(i):
                    if not _divides(i - 1, n - 1):
                        return False
            return True
        else:
            raise ValueError('The provided number must be greater than or equal to 0')

    @staticmethod
    def find_carmichael_numbers_in_range(x, y):
        if 0 <= x <= y:
            if x % 2 == 0:
                return [i for i in range(x + 1, y, 2) if carmichael.is_carmichael(i)]
            else:
                return [i for i in range(x, y, 2) if carmichael.is_carmichael(i)]
        else:
            raise ValueError('The provided range is not valid. x and y must be non-negative integers and x <= y')

    @staticmethod
    def find_first_n_carmichaels(n):
        i = 1
        carmichaels = []
        while len(carmichaels) < n:
            if carmichael.is_carmichael(i):
                carmichaels.append(i)
            i += 2
        return carmichaels