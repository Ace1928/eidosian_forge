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
class tanh(HyperbolicFunction):
    """
    ``tanh(x)`` is the hyperbolic tangent of ``x``.

    The hyperbolic tangent function is $\\frac{\\sinh(x)}{\\cosh(x)}$.

    Examples
    ========

    >>> from sympy import tanh
    >>> from sympy.abc import x
    >>> tanh(x)
    tanh(x)

    See Also
    ========

    sinh, cosh, atanh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.One - tanh(self.args[0]) ** 2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return atanh

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN
            i_coeff = _imaginary_unit_as_coefficient(arg)
            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return -I * tan(-i_coeff)
                return I * tan(i_coeff)
            elif arg.could_extract_minus_sign():
                return -cls(-arg)
            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    tanhm = tanh(m * pi * I)
                    if tanhm is S.ComplexInfinity:
                        return coth(x)
                    else:
                        return tanh(x)
            if arg.is_zero:
                return S.Zero
            if arg.func == asinh:
                x = arg.args[0]
                return x / sqrt(1 + x ** 2)
            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1) / x
            if arg.func == atanh:
                return arg.args[0]
            if arg.func == acoth:
                return 1 / arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            a = 2 ** (n + 1)
            B = bernoulli(n + 1)
            F = factorial(n + 1)
            return a * (a - 1) * B / F * x ** n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
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
        denom = sinh(re) ** 2 + cos(im) ** 2
        return (sinh(re) * cosh(re) / denom, sin(im) * cos(im) / denom)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            n = len(arg.args)
            TX = [tanh(x, evaluate=False)._eval_expand_trig() for x in arg.args]
            p = [0, 0]
            for i in range(n + 1):
                p[i % 2] += symmetric_poly(i, TX)
            return p[1] / p[0]
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul()
            if coeff.is_Integer and coeff > 1:
                T = tanh(terms)
                n = [nC(range(coeff), k) * T ** k for k in range(1, coeff + 1, 2)]
                d = [nC(range(coeff), k) * T ** k for k in range(0, coeff + 1, 2)]
                return Add(*n) / Add(*d)
        return tanh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = (exp(-arg), exp(arg))
        return (pos_exp - neg_exp) / (pos_exp + neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = (exp(-arg), exp(arg))
        return (pos_exp - neg_exp) / (pos_exp + neg_exp)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return -I * tan(I * arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        return -I / cot(I * arg)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I * sinh(arg) / sinh(pi * I / 2 - arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return I * cosh(pi * I / 2 - arg) / cosh(arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return 1 / coth(arg)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)
        if x in arg.free_symbols and Order(1, x).contains(arg):
            return arg
        else:
            return self.func(arg)

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True
        re, im = arg.as_real_imag()
        if re == 0 and im % pi == pi / 2:
            return None
        return (im % (pi / 2)).is_zero

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
        re, im = arg.as_real_imag()
        denom = cos(im) ** 2 + sinh(re) ** 2
        if denom == 0:
            return False
        elif denom.is_number:
            return True
        if arg.is_extended_real:
            return True

    def _eval_is_zero(self):
        arg = self.args[0]
        if arg.is_zero:
            return True