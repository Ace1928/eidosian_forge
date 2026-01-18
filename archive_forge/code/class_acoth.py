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
class acoth(InverseHyperbolicFunction):
    """
    ``acoth(x)`` is the inverse hyperbolic cotangent of ``x``.

    The inverse hyperbolic cotangent function.

    Examples
    ========

    >>> from sympy import acoth
    >>> from sympy.abc import x
    >>> acoth(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, coth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / (1 - self.args[0] ** 2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return pi * I / 2
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.Zero
            i_coeff = _imaginary_unit_as_coefficient(arg)
            if i_coeff is not None:
                return -I * acot(i_coeff)
            elif arg.could_extract_minus_sign():
                return -cls(-arg)
        if arg.is_zero:
            return pi * I * S.Half

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return -I * pi / 2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x ** n / n

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.ComplexInfinity:
            return (1 / arg).as_leading_term(x)
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        if x0.is_real and (1 - x0 ** 2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return self.func(x0) + I * pi
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return self.func(x0) - I * pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        if arg0.is_real and (1 - arg0 ** 2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return res + I * pi
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return res - I * pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + 1 / x) - log(1 - 1 / x)) / 2
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(1 / x)

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return pi * I / 2 * (sqrt((x - 1) / x) * sqrt(x / (x - 1)) - sqrt(1 + 1 / x) * sqrt(x / (x + 1))) + x * sqrt(1 / x ** 2) * asinh(sqrt(1 / (x ** 2 - 1)))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return coth