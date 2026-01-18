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
class acsc(InverseTrigonometricFunction):
    """
    The inverse cosecant function.

    Returns the arc cosecant of x (measured in radians).

    Explanation
    ===========

    ``acsc(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$` and for some instances when the
    result is a rational multiple of $\\pi$ (see the ``eval`` class method).

    Examples
    ========

    >>> from sympy import acsc, oo
    >>> acsc(1)
    pi/2
    >>> acsc(-1)
    -pi/2
    >>> acsc(oo)
    0
    >>> acsc(-oo) == acsc(oo)
    True
    >>> acsc(0)
    zoo

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsc

    """

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.ComplexInfinity
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.One:
                return pi / 2
            elif arg is S.NegativeOne:
                return -pi / 2
        if arg in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
            return S.Zero
        if arg.could_extract_minus_sign():
            return -cls(-arg)
        if arg.is_infinite:
            return S.Zero
        if arg.is_number:
            acsc_table = cls._acsc_table()
            if arg in acsc_table:
                return acsc_table[arg]
        if isinstance(arg, csc):
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
        if isinstance(arg, sec):
            ang = arg.args[0]
            if ang.is_comparable:
                return pi / 2 - asec(arg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1 / (self.args[0] ** 2 * sqrt(1 - 1 / self.args[0] ** 2))
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csc

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi / 2 - S.ImaginaryUnit * log(2) + S.ImaginaryUnit * log(x)
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
                return S.ImaginaryUnit * R / F * x ** n / 4

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        if x0 is S.ComplexInfinity:
            return (1 / arg).as_leading_term(x)
        if x0.is_real and (1 - x0 ** 2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return -pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = acsc(S.One + t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = S.NegativeOne + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = acsc(S.NegativeOne - t ** 2).rewrite(log).nseries(t, 0, 2 * n)
            arg1 = S.NegativeOne - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f) / f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO() * sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x ** n, x)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        if arg0.is_real and (1 - arg0 ** 2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return pi - res
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return -pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return -S.ImaginaryUnit * log(S.ImaginaryUnit / arg + sqrt(1 - 1 / arg ** 2))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return asin(1 / arg)

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return pi / 2 - acos(1 / arg)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return sqrt(x ** 2) / x * (pi / 2 - atan(sqrt(x ** 2 - 1)))

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return sqrt(arg ** 2) / arg * (pi / 2 - acot(1 / sqrt(arg ** 2 - 1)))

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi / 2 - asec(arg)