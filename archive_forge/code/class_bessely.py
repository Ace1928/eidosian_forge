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
class bessely(BesselBase):
    """
    Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $Y$ function of order $\\nu$ is defined as

    .. math ::
        Y_\\nu(z) = \\lim_{\\mu \\to \\nu} \\frac{J_\\mu(z) \\cos(\\pi \\mu)
                                            - J_{-\\mu}(z)}{\\sin(\\pi \\mu)},

    where $J_\\mu(z)$ is the Bessel function of the first kind.

    It is a solution to Bessel's equation, and linearly independent from
    $J_\\nu$.

    Examples
    ========

    >>> from sympy import bessely, yn
    >>> from sympy.abc import z, n
    >>> b = bessely(n, z)
    >>> b.diff(z)
    bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    >>> b.rewrite(yn)
    sqrt(2)*sqrt(z)*yn(n - 1/2, z)/sqrt(pi)

    See Also
    ========

    besselj, besseli, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/

    """
    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.NegativeInfinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z == I * S.Infinity:
            return exp(I * pi * (nu + 1) / 2) * S.Infinity
        if z == I * S.NegativeInfinity:
            return exp(-I * pi * (nu + 1) / 2) * S.Infinity
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return S.NegativeOne ** (-nu) * bessely(-nu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi * nu) * (cos(pi * nu) * besselj(nu, z) - besselj(-nu, z))

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(besseli)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return sqrt(2 * z / pi) * yn(nu - S.Half, self.argument)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)
        if e.is_positive:
            term_one = 2 / pi * log(z / 2) * besselj(nu, z)
            term_two = -(z / 2) ** (-nu) * factorial(nu - 1) / pi if nu.is_positive else S.Zero
            term_three = -(z / 2) ** nu / (pi * factorial(nu)) * (digamma(nu + 1) - S.EulerGamma)
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c * cdir ** e
            if not sign.is_negative:
                return sqrt(2) * (-sin(pi * nu / 2 - z + pi / 4) + 3 * cos(pi * nu / 2 - z + pi / 4) / (8 * z)) * sqrt(1 / z) / sqrt(pi)
            return self
        return super(bessely, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self
        if exp.is_positive and nu.is_integer:
            newn = ceiling(n / exp)
            bn = besselj(nu, z)
            a = (2 / pi * log(z / 2) * bn)._eval_nseries(x, n, logx, cdir)
            b, c = ([], [])
            o = Order(x ** n, x)
            r = (z / 2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r ** 2) + o).removeO()
            if nu > S.Zero:
                term = r ** (-nu) * factorial(nu - 1) / pi
                b.append(term)
                for k in range(1, nu):
                    denom = (nu - k) * k
                    if denom == S.Zero:
                        term *= t / k
                    else:
                        term *= t / denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)
            p = r ** nu / (pi * factorial(nu))
            term = p * (digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1) // 2):
                p *= -t / (k * (k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p * (digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)
            return a - Add(*b) - Add(*c)
        return super(bessely, self)._eval_nseries(x, n, logx, cdir)