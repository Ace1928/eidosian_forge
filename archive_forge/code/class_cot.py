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
class cot(TrigonometricFunction):
    """
    The cotangent function.

    Returns the cotangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cot, pi
    >>> from sympy.abc import x
    >>> cot(x**2).diff(x)
    2*x*(-cot(x**2)**2 - 1)
    >>> cot(1).diff(x)
    0
    >>> cot(pi/12)
    sqrt(3) + 2

    See Also
    ========

    sin, csc, cos, sec, tan
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cot

    """

    def period(self, symbol=None):
        return self._period(pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.NegativeOne - self ** 2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acot

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            if arg.is_zero:
                return S.ComplexInfinity
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(S.NegativeInfinity, S.Infinity)
        if arg is S.ComplexInfinity:
            return S.NaN
        if isinstance(arg, AccumBounds):
            return -tan(arg + pi / 2)
        if arg.could_extract_minus_sign():
            return -cls(-arg)
        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import coth
            return -S.ImaginaryUnit * coth(i_coeff)
        pi_coeff = _pi_coeff(arg, 2)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.ComplexInfinity
            if not pi_coeff.is_Rational:
                narg = pi_coeff * pi
                if narg != arg:
                    return cls(narg)
                return None
            if pi_coeff.is_Rational:
                if pi_coeff.q in (5, 10):
                    return tan(pi / 2 - arg)
                if pi_coeff.q > 2 and (not pi_coeff.q % 2):
                    narg = pi_coeff * pi * 2
                    cresult, sresult = (cos(narg), cos(narg - pi / 2))
                    if not isinstance(cresult, cos) and (not isinstance(sresult, cos)):
                        return 1 / sresult + cresult / sresult
                q = pi_coeff.q
                p = pi_coeff.p % q
                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    nvala, nvalb = (cls(p * pi / a), cls(p * pi / b))
                    if None in (nvala, nvalb):
                        return None
                    return (1 + nvala * nvalb) / (nvalb - nvala)
                narg = ((pi_coeff + S.Half) % 1 - S.Half) * pi
                cresult, sresult = (cos(narg), cos(narg - pi / 2))
                if not isinstance(cresult, cos) and (not isinstance(sresult, cos)):
                    if sresult == 0:
                        return S.ComplexInfinity
                    return cresult / sresult
                if narg != arg:
                    return cls(narg)
        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                cotm = cot(m * pi)
                if cotm is S.ComplexInfinity:
                    return cot(x)
                else:
                    return -tan(x)
        if arg.is_zero:
            return S.ComplexInfinity
        if isinstance(arg, acot):
            return arg.args[0]
        if isinstance(arg, atan):
            x = arg.args[0]
            return 1 / x
        if isinstance(arg, atan2):
            y, x = arg.args
            return x / y
        if isinstance(arg, asin):
            x = arg.args[0]
            return sqrt(1 - x ** 2) / x
        if isinstance(arg, acos):
            x = arg.args[0]
            return x / sqrt(1 - x ** 2)
        if isinstance(arg, acsc):
            x = arg.args[0]
            return sqrt(1 - 1 / x ** 2) * x
        if isinstance(arg, asec):
            x = arg.args[0]
            return 1 / (sqrt(1 - 1 / x ** 2) * x)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            B = bernoulli(n + 1)
            F = factorial(n + 1)
            return S.NegativeOne ** ((n + 1) // 2) * 2 ** (n + 1) * B / F * x ** n

    def _eval_nseries(self, x, n, logx, cdir=0):
        i = self.args[0].limit(x, 0) / pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return self.rewrite(tan)._eval_nseries(x, n=n, logx=logx)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        re, im = self._as_real_imag(deep=deep, **hints)
        if im:
            from sympy.functions.elementary.hyperbolic import cosh, sinh
            denom = cos(2 * re) - cosh(2 * im)
            return (-sin(2 * re) / denom, sinh(2 * im) / denom)
        else:
            return (self.func(re), S.Zero)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp)
        neg_exp, pos_exp = (exp(-arg * I), exp(arg * I))
        return I * (pos_exp + neg_exp) / (pos_exp - neg_exp)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return -I * (x ** (-I) + x ** I) / (x ** (-I) - x ** I)

    def _eval_rewrite_as_sin(self, x, **kwargs):
        return sin(2 * x) / (2 * sin(x) ** 2)

    def _eval_rewrite_as_cos(self, x, **kwargs):
        return cos(x) / cos(x - pi / 2, evaluate=False)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg) / sin(arg)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return 1 / tan(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        cos_in_sec_form = cos(arg).rewrite(sec)
        sin_in_sec_form = sin(arg).rewrite(sec)
        return cos_in_sec_form / sin_in_sec_form

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        cos_in_csc_form = cos(arg).rewrite(csc)
        sin_in_csc_form = sin(arg).rewrite(csc)
        return cos_in_csc_form / sin_in_csc_form

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        y = self.rewrite(cos).rewrite(pow)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        y = self.rewrite(cos).rewrite(sqrt)
        if y.has(cos):
            return None
        return y

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = 2 * x0 / pi
        if n.is_integer:
            lt = (arg - n * pi / 2).as_leading_term(x)
            return 1 / lt if n.is_even else -lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        x = None
        if arg.is_Add:
            n = len(arg.args)
            CX = []
            for x in arg.args:
                cx = cot(x, evaluate=False)._eval_expand_trig()
                CX.append(cx)
            Yg = numbered_symbols('Y')
            Y = [next(Yg) for i in range(n)]
            p = [0, 0]
            for i in range(n, -1, -1):
                p[(n - i) % 2] += symmetric_poly(i, Y) * (-1) ** ((n - i) % 4 // 2)
            return (p[0] / p[1]).subs(list(zip(Y, CX)))
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                I = S.ImaginaryUnit
                z = Symbol('dummy', real=True)
                P = ((z + I) ** coeff).expand()
                return (re(P) / im(P)).subs([(z, cot(terms))])
        return cot(arg)

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_real and (arg / pi).is_integer is False:
            return True
        if arg.is_imaginary:
            return True

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real and (arg / pi).is_integer is False:
            return True

    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_real and (arg / pi).is_integer is False:
            return True

    def _eval_is_zero(self):
        rest, pimult = _peeloff_pi(self.args[0])
        if pimult and rest.is_zero:
            return (pimult - S.Half).is_integer

    def _eval_subs(self, old, new):
        arg = self.args[0]
        argnew = arg.subs(old, new)
        if arg != argnew and (argnew / pi).is_integer:
            return S.ComplexInfinity
        return cot(argnew)