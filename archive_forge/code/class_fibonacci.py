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
class fibonacci(Function):
    """
    Fibonacci numbers / Fibonacci polynomials

    The Fibonacci numbers are the integer sequence defined by the
    initial terms `F_0 = 0`, `F_1 = 1` and the two-term recurrence
    relation `F_n = F_{n-1} + F_{n-2}`.  This definition
    extended to arbitrary real and complex arguments using
    the formula

    .. math :: F_z = \\frac{\\phi^z - \\cos(\\pi z) \\phi^{-z}}{\\sqrt 5}

    The Fibonacci polynomials are defined by `F_1(x) = 1`,
    `F_2(x) = x`, and `F_n(x) = x*F_{n-1}(x) + F_{n-2}(x)` for `n > 2`.
    For all positive integers `n`, `F_n(1) = F_n`.

    * ``fibonacci(n)`` gives the `n^{th}` Fibonacci number, `F_n`
    * ``fibonacci(n, x)`` gives the `n^{th}` Fibonacci polynomial in `x`, `F_n(x)`

    Examples
    ========

    >>> from sympy import fibonacci, Symbol

    >>> [fibonacci(x) for x in range(11)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    >>> fibonacci(5, Symbol('t'))
    t**4 + 3*t**2 + 1

    See Also
    ========

    bell, bernoulli, catalan, euler, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fibonacci_number
    .. [2] https://mathworld.wolfram.com/FibonacciNumber.html

    """

    @staticmethod
    def _fib(n):
        return _ifib(n)

    @staticmethod
    @recurrence_memo([None, S.One, _sym])
    def _fibpoly(n, prev):
        return (prev[-2] + _sym * prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity
        if n.is_Integer:
            if sym is None:
                n = int(n)
                if n < 0:
                    return S.NegativeOne ** (n + 1) * fibonacci(-n)
                else:
                    return Integer(cls._fib(n))
            else:
                if n < 1:
                    raise ValueError('Fibonacci polynomials are defined only for positive integer indices.')
                return cls._fibpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2 ** (-n) * sqrt(5) * ((1 + sqrt(5)) ** n - (-sqrt(5) + 1) ** n) / 5

    def _eval_rewrite_as_GoldenRatio(self, n, **kwargs):
        return (S.GoldenRatio ** n - 1 / (-S.GoldenRatio) ** n) / (2 * S.GoldenRatio - 1)