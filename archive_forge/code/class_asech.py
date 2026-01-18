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
class asech(InverseHyperbolicFunction):
    """
    ``asech(x)`` is the inverse hyperbolic secant of ``x``.

    The inverse hyperbolic secant function.

    Examples
    ========

    >>> from sympy import asech, sqrt, S
    >>> from sympy.abc import x
    >>> asech(x).diff(x)
    -1/(x*sqrt(1 - x**2))
    >>> asech(1).diff(x)
    0
    >>> asech(1)
    0
    >>> asech(S(2))
    I*pi/3
    >>> asech(-sqrt(2))
    3*I*pi/4
    >>> asech((sqrt(6) - sqrt(2)))
    I*pi/12

    See Also
    ========

    asinh, atanh, cosh, acoth

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSech/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1 / (z * sqrt(1 - z ** 2))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return pi * I / 2
            elif arg is S.NegativeInfinity:
                return pi * I / 2
            elif arg.is_zero:
                return S.Infinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi * I
        if arg.is_number:
            cst_table = _asech_table()
            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg] * I
                return cst_table[arg]
        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return I * AccumBounds(-pi / 2, pi / 2)
        if arg.is_zero:
            return S.Infinity

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1) * (n - 2)) * x ** 2 / (4 * (n // 2) ** 2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) * n
                F = factorial(k) * n // 2 * n // 2
                return -1 * R / F * x ** n / 4

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        if x0.is_negative or (1 - x0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if x0.is_positive or (x0 + 1).is_negative:
                    return -self.func(x0)
                return self.func(x0) - 2 * I * pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asech(S.One - t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asech(S.NegativeOne + t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else I * pi + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        if arg0.is_negative or (1 - arg0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if arg0.is_positive or (arg0 + 1).is_negative:
                    return -res
                return res - 2 * I * pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sech

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1 / arg + sqrt(1 / arg - 1) * sqrt(1 / arg + 1))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return acosh(1 / arg)

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return sqrt(1 / arg - 1) / sqrt(1 - 1 / arg) * (I * asinh(I / arg) + pi * S.Half)

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return I * pi * (1 - sqrt(x) * sqrt(1 / x) - I / 2 * sqrt(-x) / sqrt(x) - I / 2 * sqrt(x ** 2) / sqrt(-x ** 2)) + sqrt(1 / (x + 1)) * sqrt(x + 1) * atanh(sqrt(1 - x ** 2))

    def _eval_rewrite_as_acsch(self, x, **kwargs):
        return sqrt(1 / x - 1) / sqrt(1 - 1 / x) * (pi / 2 - I * acsch(I * x))