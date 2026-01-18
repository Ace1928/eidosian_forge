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
@staticmethod
def _bell_incomplete_poly(n, k, symbols):
    """
        The second kind of Bell polynomials (incomplete Bell polynomials).

        Calculated by recurrence formula:

        .. math:: B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1}) =
                \\sum_{m=1}^{n-k+1}
                \\x_m \\binom{n-1}{m-1} B_{n-m,k-1}(x_1, x_2, \\dotsc, x_{n-m-k})

        where
            `B_{0,0} = 1;`
            `B_{n,0} = 0; for n \\ge 1`
            `B_{0,k} = 0; for k \\ge 1`

        """
    if n == 0 and k == 0:
        return S.One
    elif n == 0 or k == 0:
        return S.Zero
    s = S.Zero
    a = S.One
    for m in range(1, n - k + 2):
        s += a * bell._bell_incomplete_poly(n - m, k - 1, symbols) * symbols[m - 1]
        a = a * (n - m) / m
    return expand_mul(s)