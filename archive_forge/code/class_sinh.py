from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
from sympy.polys.specialpolys import symmetric_poly
class sinh(HyperbolicFunction):
    """
    ``sinh(x)`` is the hyperbolic sine of ``x``.

    The hyperbolic sine function is $\\frac{e^x - e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import sinh
    >>> from sympy.abc import x
    >>> sinh(x)
    sinh(x)

    See Also
    ========

    cosh, tanh, asinh
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return cosh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return asinh

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN
            i_coeff = _imaginary_unit_as_coefficient(arg)
            if i_coeff is not None:
                return I * sin(i_coeff)
            elif arg.could_extract_minus_sign():
                return -cls(-arg)
            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m * pi * I
                    return sinh(m) * cosh(x) + cosh(m) * sinh(x)
            if arg.is_zero:
                return S.Zero
            if arg.func == asinh:
                return arg.args[0]
            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1)
            if arg.func == atanh:
                x = arg.args[0]
                return x / sqrt(1 - x ** 2)
            if arg.func == acoth:
                x = arg.args[0]
                return 1 / (sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion.
        """
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x ** 2 / (n * (n - 1))
            else:
                return x ** n / factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.
        """
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (sinh(re) * cos(im), cosh(re) * sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part * I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add:
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and (terms is not S.One):
                x = terms
                y = (coeff - 1) * x
        if x is not None:
            return (sinh(x) * cosh(y) + sinh(y) * cosh(x)).expand(trig=True)
        return sinh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return -I * sin(I * arg)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return -I / csc(I * arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I * cosh(arg + pi * I / 2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half * arg)
        return 2 * tanh_half / (1 - tanh_half ** 2)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half * arg)
        return 2 * coth_half / (coth_half ** 2 - 1)

    def _eval_rewrite_as_csch(self, arg, **kwargs):
        return 1 / csch(arg)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True
        re, im = arg.as_real_imag()
        return (im % pi).is_zero

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if rest.is_zero:
            return ipi_mult.is_integer