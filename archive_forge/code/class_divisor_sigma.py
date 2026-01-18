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
class divisor_sigma(Function):
    """
    Calculate the divisor function `\\sigma_k(n)` for positive integer n

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \\prod_{i=1}^\\omega p_i^{m_i},

    then

    .. math ::
        \\sigma_k(n) = \\prod_{i=1}^\\omega (1+p_i^k+p_i^{2k}+\\cdots
        + p_i^{m_ik}).

    Parameters
    ==========

    n : integer

    k : integer, optional
        power of divisors in the sum

        for k = 0, 1:
        ``divisor_sigma(n, 0)`` is equal to ``divisor_count(n)``
        ``divisor_sigma(n, 1)`` is equal to ``sum(divisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.ntheory import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    divisor_count, totient, divisors, factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """

    @classmethod
    def eval(cls, n, k=S.One):
        k = sympify(k)
        if n.is_prime:
            return 1 + n ** k
        if n.is_Integer:
            if n <= 0:
                raise ValueError('n must be a positive integer')
            elif k.is_Integer:
                k = int(k)
                return Integer(math.prod(((p ** (k * (e + 1)) - 1) // (p ** k - 1) if k != 0 else e + 1 for p, e in factorint(n).items())))
            else:
                return Mul(*[(p ** (k * (e + 1)) - 1) / (p ** k - 1) if k != 0 else e + 1 for p, e in factorint(n).items()])
        if n.is_integer:
            args = []
            for p, e in (_.as_base_exp() for _ in Mul.make_args(n)):
                if p.is_prime and e.is_positive:
                    args.append((p ** (k * (e + 1)) - 1) / (p ** k - 1) if k != 0 else e + 1)
                else:
                    return
            return Mul(*args)