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
class fresnels(FresnelIntegral):
    """
    Fresnel integral S.

    Explanation
    ===========

    This function is defined by

    .. math:: \\operatorname{S}(z) = \\int_0^z \\sin{\\frac{\\pi}{2} t^2} \\mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnels
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnels(0)
    0
    >>> fresnels(oo)
    1/2
    >>> fresnels(-oo)
    -1/2
    >>> fresnels(I*oo)
    -I/2
    >>> fresnels(-I*oo)
    I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnels(-z)
    -fresnels(z)
    >>> fresnels(I*z)
    -I*fresnels(z)

    The Fresnel S integral obeys the mirror symmetry
    $\\overline{S(z)} = S(\\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnels(z))
    fresnels(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnels(z), z)
    sin(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, sin, expand_func
    >>> integrate(sin(pi*z**2/2), z)
    3*fresnels(z)*gamma(3/4)/(4*gamma(7/4))
    >>> expand_func(integrate(sin(pi*z**2/2), z))
    fresnels(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnels(2).evalf(30)
    0.343415678363698242195300815958

    >>> fresnels(-2*I).evalf(30)
    0.343415678363698242195300815958*I

    See Also
    ========

    fresnelc: Fresnel cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelS
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = sin
    _sign = -S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return -pi ** 2 * x ** 4 * (4 * n - 1) / (8 * n * (2 * n + 1) * (4 * n + 3)) * p
            else:
                return x ** 3 * (-x ** 4) ** n * (S(2) ** (-2 * n - 1) * pi ** (2 * n + 1)) / ((4 * n + 3) * factorial(2 * n + 1))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One + I) / 4 * (erf((S.One + I) / 2 * sqrt(pi) * z) - I * erf((S.One - I) / 2 * sqrt(pi) * z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return pi * z ** 3 / 6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi ** 2 * z ** 4 / 16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return pi * z ** Rational(9, 4) / (sqrt(2) * (z ** 2) ** Rational(3, 4) * (-z) ** Rational(3, 4)) * meijerg([], [1], [Rational(3, 4)], [Rational(1, 4), 0], -pi ** 2 * z ** 4 / 16)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return pi * arg ** 3 / 6
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s * S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]
            p = [S.NegativeOne ** k * factorial(4 * k + 1) / (2 ** (2 * k + 2) * z ** (4 * k + 3) * 2 ** (2 * k) * factorial(2 * k)) for k in range(0, n) if 4 * k + 3 < n]
            q = [1 / (2 * z)] + [S.NegativeOne ** k * factorial(4 * k - 1) / (2 ** (2 * k + 1) * z ** (4 * k + 1) * 2 ** (2 * k - 1) * factorial(2 * k - 1)) for k in range(1, n) if 4 * k + 1 < n]
            p = [-sqrt(2 / pi) * t for t in p]
            q = [-sqrt(2 / pi) * t for t in q]
            s = 1 if point is S.Infinity else -1
            return s * S.Half + (sin(z ** 2) * Add(*p) + cos(z ** 2) * Add(*q)).subs(x, sqrt(2 / pi) * x) + Order(1 / z ** n, x)
        return super()._eval_aseries(n, args0, x, logx)