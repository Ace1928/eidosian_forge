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
class li(Function):
    """
    The classical logarithmic integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \\operatorname{li}(x) = \\int_0^x \\frac{1}{\\log(t)} \\mathrm{d}t \\,.

    Examples
    ========

    >>> from sympy import I, oo, li
    >>> from sympy.abc import z

    Several special values are known:

    >>> li(0)
    0
    >>> li(1)
    -oo
    >>> li(oo)
    oo

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(li(z), z)
    1/log(z)

    Defining the ``li`` function via an integral:
    >>> from sympy import integrate
    >>> integrate(li(z))
    z*li(z) - Ei(2*log(z))

    >>> integrate(li(z),z)
    z*li(z) - Ei(2*log(z))


    The logarithmic integral can also be defined in terms of ``Ei``:

    >>> from sympy import Ei
    >>> li(z).rewrite(Ei)
    Ei(log(z))
    >>> diff(li(z).rewrite(Ei), z)
    1/log(z)

    We can numerically evaluate the logarithmic integral to arbitrary precision
    on the whole complex plane (except the singular points):

    >>> li(2).evalf(30)
    1.04516378011749278484458888919

    >>> li(2*I).evalf(30)
    1.0652795784357498247001125598 + 3.08346052231061726610939702133*I

    We can even compute Soldner's constant by the help of mpmath:

    >>> from mpmath import findroot
    >>> findroot(li, 2)
    1.45136923488338

    Further transformations include rewriting ``li`` in terms of
    the trigonometric integrals ``Si``, ``Ci``, ``Shi`` and ``Chi``:

    >>> from sympy import Si, Ci, Shi, Chi
    >>> li(z).rewrite(Si)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Ci)
    -log(I*log(z)) - log(1/log(z))/2 + log(log(z))/2 + Ci(I*log(z)) + Shi(log(z))
    >>> li(z).rewrite(Shi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))
    >>> li(z).rewrite(Chi)
    -log(1/log(z))/2 + log(log(z))/2 + Chi(log(z)) - Shi(log(z))

    See Also
    ========

    Li: Offset logarithmic integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_integral
    .. [2] https://mathworld.wolfram.com/LogarithmicIntegral.html
    .. [3] https://dlmf.nist.gov/6
    .. [4] https://mathworld.wolfram.com/SoldnersConstant.html

    """

    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.Zero
        elif z is S.One:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        if z.is_zero:
            return S.Zero

    def fdiff(self, argindex=1):
        arg = self.args[0]
        if argindex == 1:
            return S.One / log(arg)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        z = self.args[0]
        if not z.is_extended_negative:
            return self.func(z.conjugate())

    def _eval_rewrite_as_Li(self, z, **kwargs):
        return Li(z) + li(2)

    def _eval_rewrite_as_Ei(self, z, **kwargs):
        return Ei(log(z))

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return -uppergamma(0, -log(z)) + S.Half * (log(log(z)) - log(S.One / log(z))) - log(-log(z))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        return Ci(I * log(z)) - I * Si(I * log(z)) - S.Half * (log(S.One / log(z)) - log(log(z))) - log(I * log(z))
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si

    def _eval_rewrite_as_Shi(self, z, **kwargs):
        return Chi(log(z)) - Shi(log(z)) - S.Half * (log(S.One / log(z)) - log(log(z)))
    _eval_rewrite_as_Chi = _eval_rewrite_as_Shi

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return log(z) * hyper((1, 1), (2, 2), log(z)) + S.Half * (log(log(z)) - log(S.One / log(z))) + EulerGamma

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return -log(-log(z)) - S.Half * (log(S.One / log(z)) - log(log(z))) - meijerg(((), (1,)), ((0, 0), ()), -log(z))

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return z * _eis(log(z))

    def _eval_nseries(self, x, n, logx, cdir=0):
        z = self.args[0]
        s = [log(z) ** k / (factorial(k) * k) for k in range(1, n)]
        return EulerGamma + log(log(z)) + Add(*s)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True