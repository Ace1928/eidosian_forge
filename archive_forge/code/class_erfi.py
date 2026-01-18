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
class erfi(Function):
    """
    Imaginary error function.

    Explanation
    ===========

    The function erfi is defined as:

    .. math ::
        \\mathrm{erfi}(x) = \\frac{2}{\\sqrt{\\pi}} \\int_0^x e^{t^2} \\mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfi
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfi(0)
    0
    >>> erfi(oo)
    oo
    >>> erfi(-oo)
    -oo
    >>> erfi(I*oo)
    I
    >>> erfi(-I*oo)
    -I

    In general one can pull out factors of -1 and $I$ from the argument:

    >>> erfi(-z)
    -erfi(z)

    >>> from sympy import conjugate
    >>> conjugate(erfi(z))
    erfi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfi(z), z)
    2*exp(z**2)/sqrt(pi)

    We can numerically evaluate the imaginary error function to arbitrary
    precision on the whole complex plane:

    >>> erfi(2).evalf(30)
    18.5648024145755525987042919132

    >>> erfi(-2*I).evalf(30)
    -0.995322265018952734162069256367*I

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://mathworld.wolfram.com/Erfi.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/Erfi

    """
    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 2 * exp(self.args[0] ** 2) / sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, z):
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z.is_zero:
                return S.Zero
            elif z is S.Infinity:
                return S.Infinity
        if z.is_zero:
            return S.Zero
        if z.could_extract_minus_sign():
            return -cls(-z)
        nz = z.extract_multiplicatively(I)
        if nz is not None:
            if nz is S.Infinity:
                return I
            if isinstance(nz, erfinv):
                return I * nz.args[0]
            if isinstance(nz, erfcinv):
                return I * (S.One - nz.args[0])
            if isinstance(nz, erf2inv) and nz.args[0].is_zero:
                return I * nz.args[1]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1) / S(2))
            if len(previous_terms) > 2:
                return previous_terms[-2] * x ** 2 * (n - 2) / (n * k)
            else:
                return 2 * x ** n / (n * factorial(k) * sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite('tractable', deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return -I * erf(I * z)

    def _eval_rewrite_as_erfc(self, z, **kwargs):
        return I * erfc(I * z) - I

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One + I) * z / sqrt(pi)
        return (S.One - I) * (fresnelc(arg) - I * fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One + I) * z / sqrt(pi)
        return (S.One - I) * (fresnelc(arg) - I * fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return z / sqrt(pi) * meijerg([S.Half], [], [0], [Rational(-1, 2)], -z ** 2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return 2 * z / sqrt(pi) * hyper([S.Half], [3 * S.Half], z ** 2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(-z ** 2) / z * (uppergamma(S.Half, -z ** 2) / sqrt(pi) - S.One)

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return sqrt(-z ** 2) / z - z * expint(S.Half, -z ** 2) / sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)
    as_real_imag = real_to_real_as_real_imag

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if x in arg.free_symbols and arg0.is_zero:
            return 2 * arg / sqrt(pi)
        elif arg0.is_finite:
            return self.func(arg0)
        return self.func(arg)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point is S.Infinity:
            z = self.args[0]
            s = [factorial2(2 * k - 1) / (2 ** k * z ** (2 * k + 1)) for k in range(n)] + [Order(1 / z ** n, x)]
            return -I + exp(z ** 2) / sqrt(pi) * Add(*s)
        return super(erfi, self)._eval_aseries(n, args0, x, logx)