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
class atan(InverseTrigonometricFunction):
    """
    The inverse tangent function.

    Returns the arc tangent of x (measured in radians).

    Explanation
    ===========

    ``atan(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, 0, 1, -1\\}$ and for some instances when the
    result is a rational multiple of $\\pi$ (see the eval class method).

    Examples
    ========

    >>> from sympy import atan, oo
    >>> atan(0)
    0
    >>> atan(1)
    pi/4
    >>> atan(oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan

    """
    args: tTuple[Expr]
    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / (1 + self.args[0] ** 2)
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
        return self.args[0].is_extended_positive

    def _eval_is_nonnegative(self):
        return self.args[0].is_extended_nonnegative

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return pi / 2
            elif arg is S.NegativeInfinity:
                return -pi / 2
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return pi / 4
            elif arg is S.NegativeOne:
                return -pi / 4
        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return AccumBounds(-pi / 2, pi / 2)
        if arg.could_extract_minus_sign():
            return -cls(-arg)
        if arg.is_number:
            atan_table = cls._atan_table()
            if arg in atan_table:
                return atan_table[arg]
        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import atanh
            return S.ImaginaryUnit * atanh(i_coeff)
        if arg.is_zero:
            return S.Zero
        if isinstance(arg, tan):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= pi
                if ang > pi / 2:
                    ang -= pi
                return ang
        if isinstance(arg, cot):
            ang = arg.args[0]
            if ang.is_comparable:
                ang = pi / 2 - acot(arg)
                if ang > pi / 2:
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return S.NegativeOne ** ((n - 1) // 2) * x ** n / n

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        if (1 + x0 ** 2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_negative:
                if im(x0).is_positive:
                    return self.func(x0) - pi
            elif re(ndir).is_positive:
                if im(x0).is_negative:
                    return self.func(x0) + pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg0 = self.args[0].subs(x, 0)
        if arg0 in (S.ImaginaryUnit, S.NegativeOne * S.ImaginaryUnit):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        ndir = self.args[0].dir(x, cdir if cdir else 1)
        if arg0 is S.ComplexInfinity:
            if re(ndir) > 0:
                return res - pi
            return res
        if (1 + arg0 ** 2).is_negative:
            if re(ndir).is_negative:
                if im(arg0).is_positive:
                    return res - pi
            elif re(ndir).is_positive:
                if im(arg0).is_negative:
                    return res + pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return S.ImaginaryUnit / 2 * (log(S.One - S.ImaginaryUnit * x) - log(S.One + S.ImaginaryUnit * x))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] is S.Infinity:
            return (pi / 2 - atan(1 / self.args[0]))._eval_nseries(x, n, logx)
        elif args0[0] is S.NegativeInfinity:
            return (-pi / 2 - atan(1 / self.args[0]))._eval_nseries(x, n, logx)
        else:
            return super()._eval_aseries(n, args0, x, logx)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return tan

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return sqrt(arg ** 2) / arg * (pi / 2 - asin(1 / sqrt(1 + arg ** 2)))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return sqrt(arg ** 2) / arg * acos(1 / sqrt(1 + arg ** 2))

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return acot(1 / arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return sqrt(arg ** 2) / arg * asec(sqrt(1 + arg ** 2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return sqrt(arg ** 2) / arg * (pi / 2 - acsc(sqrt(1 + arg ** 2)))