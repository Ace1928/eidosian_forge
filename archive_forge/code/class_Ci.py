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
class Ci(TrigonometricIntegral):
    """
    Cosine integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \\operatorname{Ci}(x) = \\gamma + \\log{x}
                         + \\int_0^x \\frac{\\cos{t} - 1}{t} \\mathrm{d}t
           = -\\int_x^\\infty \\frac{\\cos{t}}{t} \\mathrm{d}t,

    where $\\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \\operatorname{Ci}(z) =
        -\\frac{\\operatorname{E}_1\\left(e^{i\\pi/2} z\\right)
               + \\operatorname{E}_1\\left(e^{-i \\pi/2} z\\right)}{2}

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.

    The formula also holds as stated
    for $z \\in \\mathbb{C}$ with $\\Re(z) > 0$.
    By lifting to the principal branch, we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Ci
    >>> from sympy.abc import z

    The cosine integral is a primitive of $\\cos(z)/z$:

    >>> Ci(z).diff(z)
    cos(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Ci(z*exp_polar(2*I*pi))
    Ci(z) + 2*I*pi

    The cosine integral behaves somewhat like ordinary $\\cos$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Ci(polar_lift(I)*z)
    Chi(z) + I*pi/2
    >>> Ci(polar_lift(-1)*z)
    Ci(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Ci(z).rewrite(expint)
    -expint(1, z*exp_polar(-I*pi/2))/2 - expint(1, z*exp_polar(I*pi/2))/2

    See Also
    ========

    Si: Sine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """
    _trigfunc = cos
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Zero

    @classmethod
    def _atneginf(cls):
        return I * pi

    @classmethod
    def _minusfactor(cls, z):
        return Ci(z) + I * pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Chi(z) + I * pi / 2 * sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -(E1(polar_lift(I) * z) + E1(polar_lift(-I) * z)) / 2

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e * logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point is S.Infinity:
            z = self.args[0]
            p = [S.NegativeOne ** k * factorial(2 * k) / z ** (2 * k) for k in range(int((n - 1) / 2))] + [Order(1 / z ** n, x)]
            q = [S.NegativeOne ** k * factorial(2 * k + 1) / z ** (2 * k + 1) for k in range(int(n / 2) - 1)] + [Order(1 / z ** n, x)]
            return sin(z) / z * Add(*p) - cos(z) / z * Add(*q)
        return super(Ci, self)._eval_aseries(n, args0, x, logx)