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
class Ei(Function):
    """
    The classical exponential integral.

    Explanation
    ===========

    For use in SymPy, this function is defined as

    .. math:: \\operatorname{Ei}(x) = \\sum_{n=1}^\\infty \\frac{x^n}{n\\, n!}
                                     + \\log(x) + \\gamma,

    where $\\gamma$ is the Euler-Mascheroni constant.

    If $x$ is a polar number, this defines an analytic function on the
    Riemann surface of the logarithm. Otherwise this defines an analytic
    function in the cut plane $\\mathbb{C} \\setminus (-\\infty, 0]$.

    **Background**

    The name exponential integral comes from the following statement:

    .. math:: \\operatorname{Ei}(x) = \\int_{-\\infty}^x \\frac{e^t}{t} \\mathrm{d}t

    If the integral is interpreted as a Cauchy principal value, this statement
    holds for $x > 0$ and $\\operatorname{Ei}(x)$ as defined above.

    Examples
    ========

    >>> from sympy import Ei, polar_lift, exp_polar, I, pi
    >>> from sympy.abc import x

    >>> Ei(-1)
    Ei(-1)

    This yields a real value:

    >>> Ei(-1).n(chop=True)
    -0.219383934395520

    On the other hand the analytic continuation is not real:

    >>> Ei(polar_lift(-1)).n(chop=True)
    -0.21938393439552 + 3.14159265358979*I

    The exponential integral has a logarithmic branch point at the origin:

    >>> Ei(x*exp_polar(2*I*pi))
    Ei(x) + 2*I*pi

    Differentiation is supported:

    >>> Ei(x).diff(x)
    exp(x)/x

    The exponential integral is related to many other special functions.
    For example:

    >>> from sympy import expint, Shi
    >>> Ei(x).rewrite(expint)
    -expint(1, x*exp_polar(I*pi)) - I*pi
    >>> Ei(x).rewrite(Shi)
    Chi(x) + Shi(x)

    See Also
    ========

    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma: Upper incomplete gamma function.

    References
    ==========

    .. [1] https://dlmf.nist.gov/6.6
    .. [2] https://en.wikipedia.org/wiki/Exponential_integral
    .. [3] Abramowitz & Stegun, section 5: https://web.archive.org/web/20201128173312/http://people.math.sfu.ca/~cbm/aands/page_228.htm

    """

    @classmethod
    def eval(cls, z):
        if z.is_zero:
            return S.NegativeInfinity
        elif z is S.Infinity:
            return S.Infinity
        elif z is S.NegativeInfinity:
            return S.Zero
        if z.is_zero:
            return S.NegativeInfinity
        nz, n = z.extract_branch_factor()
        if n:
            return Ei(nz) + 2 * I * pi * n

    def fdiff(self, argindex=1):
        arg = unpolarify(self.args[0])
        if argindex == 1:
            return exp(arg) / arg
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        if (self.args[0] / polar_lift(-1)).is_positive:
            return Function._eval_evalf(self, prec) + (I * pi)._eval_evalf(prec)
        return Function._eval_evalf(self, prec)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return -uppergamma(0, polar_lift(-1) * z) - I * pi

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -expint(1, polar_lift(-1) * z) - I * pi

    def _eval_rewrite_as_li(self, z, **kwargs):
        if isinstance(z, log):
            return li(z.args[0])
        return li(exp(z))

    def _eval_rewrite_as_Si(self, z, **kwargs):
        if z.is_negative:
            return Shi(z) + Chi(z) - I * pi
        else:
            return Shi(z) + Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(z) * _eis(z)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy import re
        x0 = self.args[0].limit(x, 0)
        arg = self.args[0].as_leading_term(x, cdir=cdir)
        cdir = arg.dir(x, cdir)
        if x0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e * logx + EulerGamma - (I * pi if re(cdir).is_negative else S.Zero)
        return super()._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_Si(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point is S.Infinity:
            z = self.args[0]
            s = [factorial(k) / z ** k for k in range(n)] + [Order(1 / z ** n, x)]
            return exp(z) / z * Add(*s)
        return super(Ei, self)._eval_aseries(n, args0, x, logx)