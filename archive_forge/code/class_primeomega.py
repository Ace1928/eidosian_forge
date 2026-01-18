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
class primeomega(Function):
    """
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \\prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\\Omega(n)` is:

    .. math ::
        \\Omega(n) = \\sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """

    @classmethod
    def eval(cls, n):
        if n.is_Integer:
            if n <= 0:
                raise ValueError('n must be a positive integer')
            else:
                return sum(factorint(n).values())