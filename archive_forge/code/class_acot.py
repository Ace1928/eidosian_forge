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
class acot(InverseTrigonometricFunction):
    """
    The inverse cotangent function.

    Returns the arc cotangent of x (measured in radians).

    Explanation
    ===========

    ``acot(x)`` will evaluate automatically in the cases
    $x \\in \\{\\infty, -\\infty, \\tilde{\\infty}, 0, 1, -1\\}$
    and for some instances when the result is a rational multiple of $\\pi$
    (see the eval class method).

    A purely imaginary argument will lead to an ``acoth`` expression.

    ``acot(x)`` has a branch cut along $(-i, i)$, hence it is discontinuous
    at 0. Its range for real $x$ is $(-\\frac{\\pi}{2}, \\frac{\\pi}{2}]$.

    Examples
    ========

    >>> from sympy import acot, sqrt
    >>> acot(0)
    pi/2
    >>> acot(1)
    pi/4
    >>> acot(sqrt(3) - 2)
    -5*pi/12

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, atan2

    References
    ==========

    .. [1] https://dlmf.nist.gov/4.23
    .. [2] https://functions.wolfram.com/ElementaryFunctions/ArcCot

    """
    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1 / (1 + self.args[0] ** 2)
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
        return self.args[0].is_nonnegative

    def _eval_is_negative(self):
        return self.args[0].is_negative

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

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
                return pi / 2
            elif arg is S.One:
                return pi / 4
            elif arg is S.NegativeOne:
                return -pi / 4
        if arg is S.ComplexInfinity:
            return S.Zero
        if arg.could_extract_minus_sign():
            return -cls(-arg)
        if arg.is_number:
            atan_table = cls._atan_table()
            if arg in atan_table:
                ang = pi / 2 - atan_table[arg]
                if ang > pi / 2:
                    ang -= pi
                return ang
        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import acoth
            return -S.ImaginaryUnit * acoth(i_coeff)
        if arg.is_zero:
            return pi * S.Half
        if isinstance(arg, cot):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= pi
                if ang > pi / 2:
                    ang -= pi
                return ang
        if isinstance(arg, tan):
            ang = arg.args[0]
            if ang.is_comparable:
                ang = pi / 2 - atan(arg)
                if ang > pi / 2:
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi / 2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return S.NegativeOne ** ((n + 1) // 2) * x ** n / n

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.ComplexInfinity:
            return (1 / arg).as_leading_term(x)
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        if x0.is_imaginary and (1 + x0 ** 2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return self.func(x0) + pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return self.func(x0) - pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg0 = self.args[0].subs(x, 0)
        if arg0 in (S.ImaginaryUnit, S.NegativeOne * S.ImaginaryUnit):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        res = Function._eval_nseries(self, x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        ndir = self.args[0].dir(x, cdir if cdir else 1)
        if arg0.is_zero:
            if re(ndir) < 0:
                return res - pi
            return res
        if arg0.is_imaginary and (1 + arg0 ** 2).is_positive:
            if re(ndir).is_positive:
                if im(arg0).is_positive:
                    return res + pi
            elif re(ndir).is_negative:
                if im(arg0).is_negative:
                    return res - pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] is S.Infinity:
            return (pi / 2 - acot(1 / self.args[0]))._eval_nseries(x, n, logx)
        elif args0[0] is S.NegativeInfinity:
            return (pi * Rational(3, 2) - acot(1 / self.args[0]))._eval_nseries(x, n, logx)
        else:
            return super(atan, self)._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_log(self, x, **kwargs):
        return S.ImaginaryUnit / 2 * (log(1 - S.ImaginaryUnit / x) - log(1 + S.ImaginaryUnit / x))
    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cot

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return arg * sqrt(1 / arg ** 2) * (pi / 2 - asin(sqrt(-arg ** 2) / sqrt(-arg ** 2 - 1)))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return arg * sqrt(1 / arg ** 2) * acos(sqrt(-arg ** 2) / sqrt(-arg ** 2 - 1))

    def _eval_rewrite_as_atan(self, arg, **kwargs):
        return atan(1 / arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return arg * sqrt(1 / arg ** 2) * asec(sqrt((1 + arg ** 2) / arg ** 2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return arg * sqrt(1 / arg ** 2) * (pi / 2 - acsc(sqrt((1 + arg ** 2) / arg ** 2)))