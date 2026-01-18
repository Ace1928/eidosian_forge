from math import prod
from sympy.core import Add, S, Dummy, expand_func
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and, fuzzy_not
from sympy.core.numbers import Rational, pi, oo, I
from sympy.core.power import Pow
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.combinatorial.factorials import factorial, rf, RisingFactorial
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
class uppergamma(Function):
    """
    The upper incomplete gamma function.

    Explanation
    ===========

    It can be defined as the meromorphic continuation of

    .. math::
        \\Gamma(s, x) := \\int_x^\\infty t^{s-1} e^{-t} \\mathrm{d}t = \\Gamma(s) - \\gamma(s, x).

    where $\\gamma(s, x)$ is the lower incomplete gamma function,
    :class:`lowergamma`. This can be shown to be the same as

    .. math::
        \\Gamma(s, x) = \\Gamma(s) - \\frac{x^s}{s} {}_1F_1\\left({s \\atop s+1} \\middle| -x\\right),

    where ${}_1F_1$ is the (confluent) hypergeometric function.

    The upper incomplete gamma function is also essentially equivalent to the
    generalized exponential integral:

    .. math::
        \\operatorname{E}_{n}(x) = \\int_{1}^{\\infty}{\\frac{e^{-xt}}{t^n} \\, dt} = x^{n-1}\\Gamma(1-n,x).

    Examples
    ========

    >>> from sympy import uppergamma, S
    >>> from sympy.abc import s, x
    >>> uppergamma(s, x)
    uppergamma(s, x)
    >>> uppergamma(3, x)
    2*(x**2/2 + x + 1)*exp(-x)
    >>> uppergamma(-S(1)/2, x)
    -2*sqrt(pi)*erfc(sqrt(x)) + 2*exp(-x)/sqrt(x)
    >>> uppergamma(-2, x)
    expint(3, x)/x**2

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function#Upper_incomplete_gamma_function
    .. [2] Abramowitz, Milton; Stegun, Irene A., eds. (1965), Chapter 6,
           Section 5, Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables
    .. [3] https://dlmf.nist.gov/8
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma2/
    .. [5] https://functions.wolfram.com/GammaBetaErf/Gamma3/
    .. [6] https://en.wikipedia.org/wiki/Exponential_integral#Relation_with_other_functions

    """

    def fdiff(self, argindex=2):
        from sympy.functions.special.hyper import meijerg
        if argindex == 2:
            a, z = self.args
            return -exp(-unpolarify(z)) * z ** (a - 1)
        elif argindex == 1:
            a, z = self.args
            return uppergamma(a, z) * log(z) + meijerg([], [1, 1], [0, 0, a], [], z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        if all((x.is_number for x in self.args)):
            a = self.args[0]._to_mpmath(prec)
            z = self.args[1]._to_mpmath(prec)
            with workprec(prec):
                res = mp.gammainc(a, z, mp.inf)
            return Expr._from_mpmath(res, prec)
        return self

    @classmethod
    def eval(cls, a, z):
        from sympy.functions.special.error_functions import expint
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z is oo:
                return S.Zero
            elif z.is_zero:
                if re(a).is_positive:
                    return gamma(a)
        nx, n = z.extract_branch_factor()
        if a.is_integer and a.is_positive:
            nx = unpolarify(z)
            if z != nx:
                return uppergamma(a, nx)
        elif a.is_integer and a.is_nonpositive:
            if n != 0:
                return -2 * pi * I * n * S.NegativeOne ** (-a) / factorial(-a) + uppergamma(a, nx)
        elif n != 0:
            return gamma(a) * (1 - exp(2 * pi * I * n * a)) + exp(2 * pi * I * n * a) * uppergamma(a, nx)
        if a.is_Number:
            if a is S.Zero and z.is_positive:
                return -Ei(-z)
            elif a is S.One:
                return exp(-z)
            elif a is S.Half:
                return sqrt(pi) * erfc(sqrt(z))
            elif a.is_Integer or (2 * a).is_Integer:
                b = a - 1
                if b.is_positive:
                    if a.is_integer:
                        return exp(-z) * factorial(b) * Add(*[z ** k / factorial(k) for k in range(a)])
                    else:
                        return gamma(a) * erfc(sqrt(z)) + S.NegativeOne ** (a - S(3) / 2) * exp(-z) * sqrt(z) * Add(*[gamma(-S.Half - k) * (-z) ** k / gamma(1 - a) for k in range(a - S.Half)])
                elif b.is_Integer:
                    return expint(-b, z) * unpolarify(z) ** (b + 1)
                if not a.is_Integer:
                    return S.NegativeOne ** (S.Half - a) * pi * erfc(sqrt(z)) / gamma(1 - a) - z ** a * exp(-z) * Add(*[z ** k * gamma(a) / gamma(a + k + 1) for k in range(S.Half - a)])
        if a.is_zero and z.is_positive:
            return -Ei(-z)
        if z.is_zero and re(a).is_positive:
            return gamma(a)

    def _eval_conjugate(self):
        z = self.args[1]
        if z not in (S.Zero, S.NegativeInfinity):
            return self.func(self.args[0].conjugate(), z.conjugate())

    def _eval_is_meromorphic(self, x, a):
        return lowergamma._eval_is_meromorphic(self, x, a)

    def _eval_rewrite_as_lowergamma(self, s, x, **kwargs):
        return gamma(s) - lowergamma(s, x)

    def _eval_rewrite_as_tractable(self, s, x, **kwargs):
        return exp(loggamma(s)) - lowergamma(s, x)

    def _eval_rewrite_as_expint(self, s, x, **kwargs):
        from sympy.functions.special.error_functions import expint
        return expint(1 - s, x) * x ** s