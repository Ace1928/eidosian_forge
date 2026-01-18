from sympy.core import EulerGamma  # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, expand_mul
from sympy.core.numbers import I, pi, Rational
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg
class erf(Function):
    """
    The Gauss error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \\mathrm{erf}(x) = \\frac{2}{\\sqrt{\\pi}} \\int_0^x e^{-t^2} \\mathrm{d}t.

    Examples
    ========

    >>> from sympy import I, oo, erf
    >>> from sympy.abc import z

    Several special values are known:

    >>> erf(0)
    0
    >>> erf(oo)
    1
    >>> erf(-oo)
    -1
    >>> erf(I*oo)
    oo*I
    >>> erf(-I*oo)
    -oo*I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erf(-z)
    -erf(z)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf(z))
    erf(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erf(z), z)
    2*exp(-z**2)/sqrt(pi)

    We can numerically evaluate the error function to arbitrary precision
    on the whole complex plane:

    >>> erf(4).evalf(30)
    0.999999984582742099719981147840

    >>> erf(-4*I).evalf(30)
    -1296959.73071763923152794095062*I

    See Also
    ========

    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erf.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erf

    """
    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2 * exp(-self.args[0] ** 2) / sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfinv

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
        if isinstance(arg, erfinv):
            return arg.args[0]
        if isinstance(arg, erfcinv):
            return S.One - arg.args[0]
        if arg.is_zero:
            return S.Zero
        if isinstance(arg, erf2inv) and arg.args[0].is_zero:
            return arg.args[1]
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return arg
        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1) / S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x ** 2 * (n - 2) / (n * k)
            else:
                return 2 * S.NegativeOne ** k * x ** n / (n * factorial(k) * sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True
        else:
            return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(z ** 2) / z * (S.One - uppergamma(S.Half, z ** 2) / sqrt(pi))

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I) * z / sqrt(pi)
        return (S.One + I) * (fresnelc(arg) - I * fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One - I) * z / sqrt(pi)
        return (S.One + I) * (fresnelc(arg) - I * fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z / sqrt(pi) * meijerg([S.Half], [], [0], [Rational(-1, 2)], z ** 2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2 * z / sqrt(pi) * hyper([S.Half], [3 * S.Half], -z ** 2)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(z ** 2) / z - z * expint(S.Half, z ** 2) / sqrt(pi)

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        from sympy.series.limits import limit
        if limitvar:
            lim = limit(z, limitvar, S.Infinity)
            if lim is S.NegativeInfinity:
                return S.NegativeOne + _erfs(-z) * exp(-z ** 2)
        return S.One - _erfs(z) * exp(-z ** 2)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return S.One - erfc(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return -I * erfi(I * z)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if x in arg.free_symbols and arg0.is_zero:
            return 2 * arg / sqrt(pi)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point in [S.Infinity, S.NegativeInfinity]:
            z = self.args[0]
            try:
                _, ex = z.leadterm(x)
            except (ValueError, NotImplementedError):
                return self
            ex = -ex
            if ex.is_positive:
                newn = ceiling(n / ex)
                s = [S.NegativeOne ** k * factorial2(2 * k - 1) / (z ** (2 * k + 1) * 2 ** k) for k in range(newn)] + [Order(1 / z ** newn, x)]
                return S.One - exp(-z ** 2) / sqrt(pi) * Add(*s)
        return super(erf, self)._eval_aseries(n, args0, x, logx)
    as_real_imag = real_to_real_as_real_imag