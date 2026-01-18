from functools import wraps
from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn
from mpmath import mp, workprec
class marcumq(Function):
    """
    The Marcum Q-function.

    Explanation
    ===========

    The Marcum Q-function is defined by the meromorphic continuation of

    .. math::
        Q_m(a, b) = a^{- m + 1} \\int_{b}^{\\infty} x^{m} e^{- \\frac{a^{2}}{2} - \\frac{x^{2}}{2}} I_{m - 1}\\left(a x\\right)\\, dx

    Examples
    ========

    >>> from sympy import marcumq
    >>> from sympy.abc import m, a, b
    >>> marcumq(m, a, b)
    marcumq(m, a, b)

    Special values:

    >>> marcumq(m, 0, b)
    uppergamma(m, b**2/2)/gamma(m)
    >>> marcumq(0, 0, 0)
    0
    >>> marcumq(0, a, 0)
    1 - exp(-a**2/2)
    >>> marcumq(1, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2
    >>> marcumq(2, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    Differentiation with respect to $a$ and $b$ is supported:

    >>> from sympy import diff
    >>> diff(marcumq(m, a, b), a)
    a*(-marcumq(m, a, b) + marcumq(m + 1, a, b))
    >>> diff(marcumq(m, a, b), b)
    -a**(1 - m)*b**m*exp(-a**2/2 - b**2/2)*besseli(m - 1, a*b)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Marcum_Q-function
    .. [2] https://mathworld.wolfram.com/MarcumQ-Function.html

    """

    @classmethod
    def eval(cls, m, a, b):
        if a is S.Zero:
            if m is S.Zero and b is S.Zero:
                return S.Zero
            return uppergamma(m, b ** 2 * S.Half) / gamma(m)
        if m is S.Zero and b is S.Zero:
            return 1 - 1 / exp(a ** 2 * S.Half)
        if a == b:
            if m is S.One:
                return (1 + exp(-a ** 2) * besseli(0, a ** 2)) * S.Half
            if m == 2:
                return S.Half + S.Half * exp(-a ** 2) * besseli(0, a ** 2) + exp(-a ** 2) * besseli(1, a ** 2)
        if a.is_zero:
            if m.is_zero and b.is_zero:
                return S.Zero
            return uppergamma(m, b ** 2 * S.Half) / gamma(m)
        if m.is_zero and b.is_zero:
            return 1 - 1 / exp(a ** 2 * S.Half)

    def fdiff(self, argindex=2):
        m, a, b = self.args
        if argindex == 2:
            return a * (-marcumq(m, a, b) + marcumq(1 + m, a, b))
        elif argindex == 3:
            return -b ** m / a ** (m - 1) * exp(-(a ** 2 + b ** 2) / 2) * besseli(m - 1, a * b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, m, a, b, **kwargs):
        from sympy.integrals.integrals import Integral
        x = kwargs.get('x', Dummy('x'))
        return a ** (1 - m) * Integral(x ** m * exp(-(x ** 2 + a ** 2) / 2) * besseli(m - 1, a * x), [x, b, S.Infinity])

    def _eval_rewrite_as_Sum(self, m, a, b, **kwargs):
        from sympy.concrete.summations import Sum
        k = kwargs.get('k', Dummy('k'))
        return exp(-(a ** 2 + b ** 2) / 2) * Sum((a / b) ** k * besseli(k, a * b), [k, 1 - m, S.Infinity])

    def _eval_rewrite_as_besseli(self, m, a, b, **kwargs):
        if a == b:
            if m == 1:
                return (1 + exp(-a ** 2) * besseli(0, a ** 2)) / 2
            if m.is_Integer and m >= 2:
                s = sum([besseli(i, a ** 2) for i in range(1, m)])
                return S.Half + exp(-a ** 2) * besseli(0, a ** 2) / 2 + exp(-a ** 2) * s

    def _eval_is_zero(self):
        if all((arg.is_zero for arg in self.args)):
            return True