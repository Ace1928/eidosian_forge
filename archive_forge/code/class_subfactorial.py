from __future__ import annotations
from functools import reduce
from sympy.core import S, sympify, Dummy, Mod
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
from sympy.core.numbers import Integer, pi, I
from sympy.core.relational import Eq
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.ntheory import sieve
from sympy.polys.polytools import Poly
from math import factorial as _factorial, prod, sqrt as _sqrt
class subfactorial(CombinatorialFunction):
    """The subfactorial counts the derangements of $n$ items and is
    defined for non-negative integers as:

    .. math:: !n = \\begin{cases} 1 & n = 0 \\\\ 0 & n = 1 \\\\
                    (n-1)(!(n-1) + !(n-2)) & n > 1 \\end{cases}

    It can also be written as ``int(round(n!/exp(1)))`` but the
    recursive definition with caching is implemented for this function.

    An interesting analytic expression is the following [2]_

    .. math:: !x = \\Gamma(x + 1, -1)/e

    which is valid for non-negative integers `x`. The above formula
    is not very useful in case of non-integers. `\\Gamma(x + 1, -1)` is
    single-valued only for integral arguments `x`, elsewhere on the positive
    real axis it has an infinite number of branches none of which are real.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Subfactorial
    .. [2] https://mathworld.wolfram.com/Subfactorial.html

    Examples
    ========

    >>> from sympy import subfactorial
    >>> from sympy.abc import n
    >>> subfactorial(n + 1)
    subfactorial(n + 1)
    >>> subfactorial(5)
    44

    See Also
    ========

    factorial, uppergamma,
    sympy.utilities.iterables.generate_derangements
    """

    @classmethod
    @cacheit
    def _eval(self, n):
        if not n:
            return S.One
        elif n == 1:
            return S.Zero
        else:
            z1, z2 = (1, 0)
            for i in range(2, n + 1):
                z1, z2 = (z2, (i - 1) * (z2 + z1))
            return z2

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg.is_Integer and arg.is_nonnegative:
                return cls._eval(arg)
            elif arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity

    def _eval_is_even(self):
        if self.args[0].is_odd and self.args[0].is_nonnegative:
            return True

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_rewrite_as_factorial(self, arg, **kwargs):
        from sympy.concrete.summations import summation
        i = Dummy('i')
        f = S.NegativeOne ** i / factorial(i)
        return factorial(arg) * summation(f, (i, 0, arg))

    def _eval_rewrite_as_gamma(self, arg, piecewise=True, **kwargs):
        from sympy.functions.elementary.exponential import exp
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        return (S.NegativeOne ** (arg + 1) * exp(-I * pi * arg) * lowergamma(arg + 1, -1) + gamma(arg + 1)) * exp(-1)

    def _eval_rewrite_as_uppergamma(self, arg, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return uppergamma(arg + 1, -1) / S.Exp1

    def _eval_is_nonnegative(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_odd(self):
        if self.args[0].is_even and self.args[0].is_nonnegative:
            return True