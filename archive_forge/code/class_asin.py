from typing import Tuple as tTuple, Union as tUnion
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.relational import Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.combinatorial.numbers import bernoulli, euler
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary._trigonometric_special import (
from sympy.logic.boolalg import And
from sympy.ntheory import factorint
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.iterables import numbered_symbols
class asin(InverseTrigonometricFunction):
    """
    The inverse sine function.

    Returns the arcsine of x in radians.

    Explanation
    ===========

    ``asin(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$ and for some instances when the
    result is a rational multiple of $\\pi$ (see the ``eval`` class method).

    A purely imaginary argument will lead to an asinh expression.

    Examples
    ========

    >>> from sympy import asin, oo
    >>> asin(1)
    pi/2
    >>> asin(-1)
    -pi/2
    >>> asin(-oo)
    oo*I
    >>> asin(oo)
    -oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSin

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / sqrt(1 - self.args[0] ** 2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    def _eval_is_positive(self):
        return self._eval_is_extended_real() and self.args[0].is_positive

    def _eval_is_negative(self):
        return self._eval_is_extended_real() and self.args[0].is_negative

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.NegativeInfinity * S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.Infinity * S.ImaginaryUnit
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return pi / 2
            elif arg is S.NegativeOne:
                return -pi / 2
        if arg is S.ComplexInfinity:
            return S.ComplexInfinity
        if arg.could_extract_minus_sign():
            return -cls(-arg)
        if arg.is_number:
            asin_table = cls._asin_table()
            if arg in asin_table:
                return asin_table[arg]
        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import asinh
            return S.ImaginaryUnit * asinh(i_coeff)
        if arg.is_zero:
            return S.Zero
        if isinstance(arg, sin):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= 2 * pi
                if ang > pi:
                    ang = pi - ang
                if ang > pi / 2:
                    ang = pi - ang
                if ang < -pi / 2:
                    ang = -pi - ang
                return ang
        if isinstance(arg, cos):
            ang = arg.args[0]
            if ang.is_comparable:
                return pi / 2 - acos(arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
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
                return R / F * x ** n / n

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        if (1 - x0 ** 2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return -pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asin(S.One - t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else pi / 2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asin(S.NegativeOne + t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            if not g.is_meromorphic(x, 0):
                return O(1) if n == 0 else -pi / 2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        if (1 - arg0 ** 2).is_negative:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return -pi - res
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return pi / 2 - acos(x)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return 2 * atan(x / (1 + sqrt(1 - x ** 2)))

    def _eval_rewrite_as_log(self, x, **kwargs):
        return -S.ImaginaryUnit * log(S.ImaginaryUnit * x + sqrt(1 - x ** 2))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return 2 * acot((1 + sqrt(1 - arg ** 2)) / arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi / 2 - asec(1 / arg)

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return acsc(1 / arg)

    def _eval_is_extended_real(self):
        x = self.args[0]
        return x.is_extended_real and (1 - abs(x)).is_nonnegative

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sin