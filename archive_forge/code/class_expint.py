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
class expint(Function):
    """
    Generalized exponential integral.

    Explanation
    ===========

    This function is defined as

    .. math:: \\operatorname{E}_\\nu(z) = z^{\\nu - 1} \\Gamma(1 - \\nu, z),

    where $\\Gamma(1 - \\nu, z)$ is the upper incomplete gamma function
    (``uppergamma``).

    Hence for $z$ with positive real part we have

    .. math:: \\operatorname{E}_\\nu(z)
              =   \\int_1^\\infty \\frac{e^{-zt}}{t^\\nu} \\mathrm{d}t,

    which explains the name.

    The representation as an incomplete gamma function provides an analytic
    continuation for $\\operatorname{E}_\\nu(z)$. If $\\nu$ is a
    non-positive integer, the exponential integral is thus an unbranched
    function of $z$, otherwise there is a branch point at the origin.
    Refer to the incomplete gamma function documentation for details of the
    branching behavior.

    Examples
    ========

    >>> from sympy import expint, S
    >>> from sympy.abc import nu, z

    Differentiation is supported. Differentiation with respect to $z$ further
    explains the name: for integral orders, the exponential integral is an
    iterated integral of the exponential function.

    >>> expint(nu, z).diff(z)
    -expint(nu - 1, z)

    Differentiation with respect to $\\nu$ has no classical expression:

    >>> expint(nu, z).diff(nu)
    -z**(nu - 1)*meijerg(((), (1, 1)), ((0, 0, 1 - nu), ()), z)

    At non-postive integer orders, the exponential integral reduces to the
    exponential function:

    >>> expint(0, z)
    exp(-z)/z
    >>> expint(-1, z)
    exp(-z)/z + exp(-z)/z**2

    At half-integers it reduces to error functions:

    >>> expint(S(1)/2, z)
    sqrt(pi)*erfc(sqrt(z))/sqrt(z)

    At positive integer orders it can be rewritten in terms of exponentials
    and ``expint(1, z)``. Use ``expand_func()`` to do this:

    >>> from sympy import expand_func
    >>> expand_func(expint(5, z))
    z**4*expint(1, z)/24 + (-z**3 + z**2 - 2*z + 6)*exp(-z)/24

    The generalised exponential integral is essentially equivalent to the
    incomplete gamma function:

    >>> from sympy import uppergamma
    >>> expint(nu, z).rewrite(uppergamma)
    z**(nu - 1)*uppergamma(1 - nu, z)

    As such it is branched at the origin:

    >>> from sympy import exp_polar, pi, I
    >>> expint(4, z*exp_polar(2*pi*I))
    I*pi*z**3/3 + expint(4, z)
    >>> expint(nu, z*exp_polar(2*pi*I))
    z**(nu - 1)*(exp(2*I*pi*nu) - 1)*gamma(1 - nu) + expint(nu, z)

    See Also
    ========

    Ei: Another related function called exponential integral.
    E1: The classical case, returns expint(1, z).
    li: Logarithmic integral.
    Li: Offset logarithmic integral.
    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    uppergamma

    References
    ==========

    .. [1] https://dlmf.nist.gov/8.19
    .. [2] https://functions.wolfram.com/GammaBetaErf/ExpIntegralE/
    .. [3] https://en.wikipedia.org/wiki/Exponential_integral

    """

    @classmethod
    def eval(cls, nu, z):
        from sympy.functions.special.gamma_functions import gamma, uppergamma
        nu2 = unpolarify(nu)
        if nu != nu2:
            return expint(nu2, z)
        if nu.is_Integer and nu <= 0 or (not nu.is_Integer and (2 * nu).is_Integer):
            return unpolarify(expand_mul(z ** (nu - 1) * uppergamma(1 - nu, z)))
        z, n = z.extract_branch_factor()
        if n is S.Zero:
            return
        if nu.is_integer:
            if not nu > 0:
                return
            return expint(nu, z) - 2 * pi * I * n * S.NegativeOne ** (nu - 1) / factorial(nu - 1) * unpolarify(z) ** (nu - 1)
        else:
            return (exp(2 * I * pi * nu * n) - 1) * z ** (nu - 1) * gamma(1 - nu) + expint(nu, z)

    def fdiff(self, argindex):
        nu, z = self.args
        if argindex == 1:
            return -z ** (nu - 1) * meijerg([], [1, 1], [0, 0, 1 - nu], [], z)
        elif argindex == 2:
            return -expint(nu - 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_uppergamma(self, nu, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return z ** (nu - 1) * uppergamma(1 - nu, z)

    def _eval_rewrite_as_Ei(self, nu, z, **kwargs):
        if nu == 1:
            return -Ei(z * exp_polar(-I * pi)) - I * pi
        elif nu.is_Integer and nu > 1:
            x = -unpolarify(z)
            return x ** (nu - 1) / factorial(nu - 1) * E1(z).rewrite(Ei) + exp(x) / factorial(nu - 1) * Add(*[factorial(nu - k - 2) * x ** k for k in range(nu - 1)])
        else:
            return self

    def _eval_expand_func(self, **hints):
        return self.rewrite(Ei).rewrite(expint, **hints)

    def _eval_rewrite_as_Si(self, nu, z, **kwargs):
        if nu != 1:
            return self
        return Shi(z) - Chi(z)
    _eval_rewrite_as_Ci = _eval_rewrite_as_Si
    _eval_rewrite_as_Chi = _eval_rewrite_as_Si
    _eval_rewrite_as_Shi = _eval_rewrite_as_Si

    def _eval_nseries(self, x, n, logx, cdir=0):
        if not self.args[0].has(x):
            nu = self.args[0]
            if nu == 1:
                f = self._eval_rewrite_as_Si(*self.args)
                return f._eval_nseries(x, n, logx)
            elif nu.is_Integer and nu > 1:
                f = self._eval_rewrite_as_Ei(*self.args)
                return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[1]
        nu = self.args[0]
        if point is S.Infinity:
            z = self.args[1]
            s = [S.NegativeOne ** k * RisingFactorial(nu, k) / z ** k for k in range(n)] + [Order(1 / z ** n, x)]
            return exp(-z) / z * Add(*s)
        return super(expint, self)._eval_aseries(n, args0, x, logx)