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
class partition(Function):
    """
    Partition numbers

    The Partition numbers are a sequence of integers `p_n` that represent the
    number of distinct ways of representing `n` as a sum of natural numbers
    (with order irrelevant). The generating function for `p_n` is given by:

    .. math:: \\sum_{n=0}^\\infty p_n x^n = \\prod_{k=1}^\\infty (1 - x^k)^{-1}

    Examples
    ========

    >>> from sympy import partition, Symbol
    >>> [partition(n) for n in range(9)]
    [1, 1, 2, 3, 5, 7, 11, 15, 22]
    >>> n = Symbol('n', integer=True, negative=True)
    >>> partition(n)
    0

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_(number_theory%29
    .. [2] https://en.wikipedia.org/wiki/Pentagonal_number_theorem

    """

    @staticmethod
    def _partition(n):
        L = len(_npartition)
        if n < L:
            return _npartition[n]
        for _n in range(L, n + 1):
            v, p, i = (0, 0, 0)
            while 1:
                s = 0
                p += 3 * i + 1
                if _n >= p:
                    s += _npartition[_n - p]
                i += 1
                gp = p + i
                if _n >= gp:
                    s += _npartition[_n - gp]
                if s == 0:
                    break
                else:
                    v += s if i % 2 == 1 else -s
            _npartition.append(v)
        return v

    @classmethod
    def eval(cls, n):
        is_int = n.is_integer
        if is_int == False:
            raise ValueError('Partition numbers are defined only for integers')
        elif is_int:
            if n.is_negative:
                return S.Zero
            if n.is_zero or (n - 1).is_zero:
                return S.One
            if n.is_Integer:
                return Integer(cls._partition(n))

    def _eval_is_integer(self):
        if self.args[0].is_integer:
            return True

    def _eval_is_negative(self):
        if self.args[0].is_integer:
            return False

    def _eval_is_positive(self):
        n = self.args[0]
        if n.is_nonnegative and n.is_integer:
            return True