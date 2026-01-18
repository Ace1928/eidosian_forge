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
class acosh(InverseHyperbolicFunction):
    """
    ``acosh(x)`` is the inverse hyperbolic cosine of ``x``.

    The inverse hyperbolic cosine function.

    Examples
    ========

    >>> from sympy import acosh
    >>> from sympy.abc import x
    >>> acosh(x).diff(x)
    1/(sqrt(x - 1)*sqrt(x + 1))
    >>> acosh(1)
    0

    See Also
    ========

    asinh, atanh, cosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            arg = self.args[0]
            return 1 / (sqrt(arg - 1) * sqrt(arg + 1))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return pi * I / 2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi * I
        if arg.is_number:
            cst_table = _acosh_table()
            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg] * I
                return cst_table[arg]
        if arg is S.ComplexInfinity:
            return S.ComplexInfinity
        if arg == I * S.Infinity:
            return S.Infinity + I * pi / 2
        if arg == -I * S.Infinity:
            return S.Infinity - I * pi / 2
        if arg.is_zero:
            return pi * I * S.Half
        if isinstance(arg, cosh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return Abs(z)
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(i / pi)
                m = z - I * pi * f
                even = f.is_even
                if even is True:
                    if r.is_nonnegative:
                        return m
                    elif r.is_negative:
                        return -m
                elif even is False:
                    m -= I * pi
                    if r.is_nonpositive:
                        return -m
                    elif r.is_positive:
                        return m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return I * pi / 2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p * (n - 2) ** 2 / (n * (n - 1)) * x ** 2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R / F * I * x ** n / n

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        if (x0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (x0 + 1).is_negative:
                    return self.func(x0) - 2 * I * pi
                return -self.func(x0)
            elif not im(ndir).is_positive:
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
        if (arg0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (arg0 + 1).is_negative:
                    return res - 2 * I * pi
                return -res
            elif not im(ndir).is_positive:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x + 1) * sqrt(x - 1))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return sqrt(x - 1) / sqrt(1 - x) * acos(x)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return sqrt(x - 1) / sqrt(1 - x) * (pi / 2 - asin(x))

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return sqrt(x - 1) / sqrt(1 - x) * (pi / 2 + I * asinh(I * x))

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        sxm1 = sqrt(x - 1)
        s1mx = sqrt(1 - x)
        sx2m1 = sqrt(x ** 2 - 1)
        return pi / 2 * sxm1 / s1mx * (1 - x * sqrt(1 / x ** 2)) + sxm1 * sqrt(x + 1) / sx2m1 * atanh(sx2m1 / x)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cosh

    def _eval_is_zero(self):
        if (self.args[0] - 1).is_zero:
            return True