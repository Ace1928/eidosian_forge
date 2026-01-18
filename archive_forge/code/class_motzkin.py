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
class motzkin(Function):
    """
    The nth Motzkin number is the number
    of ways of drawing non-intersecting chords
    between n points on a circle (not necessarily touching
    every point by a chord). The Motzkin numbers are named
    after Theodore Motzkin and have diverse applications
    in geometry, combinatorics and number theory.

    Motzkin numbers are the integer sequence defined by the
    initial terms `M_0 = 1`, `M_1 = 1` and the two-term recurrence relation
    `M_n = \x0crac{2*n + 1}{n + 2} * M_{n-1} + \x0crac{3n - 3}{n + 2} * M_{n-2}`.


    Examples
    ========

    >>> from sympy import motzkin

    >>> motzkin.is_motzkin(5)
    False
    >>> motzkin.find_motzkin_numbers_in_range(2,300)
    [2, 4, 9, 21, 51, 127]
    >>> motzkin.find_motzkin_numbers_in_range(2,900)
    [2, 4, 9, 21, 51, 127, 323, 835]
    >>> motzkin.find_first_n_motzkins(10)
    [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Motzkin_number
    .. [2] https://mathworld.wolfram.com/MotzkinNumber.html

    """

    @staticmethod
    def is_motzkin(n):
        try:
            n = as_int(n)
        except ValueError:
            return False
        if n > 0:
            if n in (1, 2):
                return True
            tn1 = 1
            tn = 2
            i = 3
            while tn < n:
                a = ((2 * i + 1) * tn + (3 * i - 3) * tn1) / (i + 2)
                i += 1
                tn1 = tn
                tn = a
            if tn == n:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def find_motzkin_numbers_in_range(x, y):
        if 0 <= x <= y:
            motzkins = []
            if x <= 1 <= y:
                motzkins.append(1)
            tn1 = 1
            tn = 2
            i = 3
            while tn <= y:
                if tn >= x:
                    motzkins.append(tn)
                a = ((2 * i + 1) * tn + (3 * i - 3) * tn1) / (i + 2)
                i += 1
                tn1 = tn
                tn = int(a)
            return motzkins
        else:
            raise ValueError('The provided range is not valid. This condition should satisfy x <= y')

    @staticmethod
    def find_first_n_motzkins(n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        motzkins = [1]
        if n >= 1:
            motzkins.append(1)
        tn1 = 1
        tn = 2
        i = 3
        while i <= n:
            motzkins.append(tn)
            a = ((2 * i + 1) * tn + (3 * i - 3) * tn1) / (i + 2)
            i += 1
            tn1 = tn
            tn = int(a)
        return motzkins

    @staticmethod
    @recurrence_memo([S.One, S.One])
    def _motzkin(n, prev):
        return ((2 * n + 1) * prev[-1] + (3 * n - 3) * prev[-2]) // (n + 2)

    @classmethod
    def eval(cls, n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        return Integer(cls._motzkin(n - 1))