from functools import wraps
from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn
from mpmath import mp, workprec
class airybi(AiryBase):
    """
    The Airy function $\\operatorname{Bi}$ of the second kind.

    Explanation
    ===========

    The Airy function $\\operatorname{Bi}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \\frac{\\mathrm{d}^2 w(z)}{\\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \\operatorname{Bi}(z) := \\frac{1}{\\pi}
                 \\int_0^\\infty
                   \\exp\\left(-\\frac{t^3}{3} + z t\\right)
                   + \\sin\\left(\\frac{t^3}{3} + z t\\right) \\mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybi
    >>> from sympy.abc import z

    >>> airybi(z)
    airybi(z)

    Several special values are known:

    >>> airybi(0)
    3**(5/6)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airybi(oo)
    oo
    >>> airybi(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybi(z))
    airybi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybi(z), z)
    airybiprime(z)
    >>> diff(airybi(z), z, 2)
    z*airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybi(z), z, 0, 3)
    3**(1/3)*gamma(1/3)/(2*pi) + 3**(2/3)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybi(-2).evalf(50)
    -0.41230258795639848808323405461146104203453483447240

    Rewrite $\\operatorname{Bi}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybi(z).rewrite(hyper)
    3**(1/6)*z*hyper((), (4/3,), z**3/9)/gamma(1/3) + 3**(5/6)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """
    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.One / (3 ** Rational(1, 6) * gamma(Rational(2, 3)))
        if arg.is_zero:
            return S.One / (3 ** Rational(1, 6) * gamma(Rational(2, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return airybiprime(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return cbrt(3) * x * Abs(sin(Rational(2, 3) * pi * (n + S.One))) * factorial((n - S.One) / S(3)) / ((n + S.One) * Abs(cos(Rational(2, 3) * pi * (n + S.Half))) * factorial((n - 2) / S(3))) * p
            else:
                return S.One / (root(3, 6) * pi) * gamma((n + S.One) / S(3)) * Abs(sin(Rational(2, 3) * pi * (n + S.One))) / factorial(n) * (cbrt(3) * x) ** n

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return sqrt(-z / 3) * (besselj(-ot, tt * a) - besselj(ot, tt * a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return sqrt(z) / sqrt(3) * (besseli(-ot, tt * a) + besseli(ot, tt * a))
        else:
            b = Pow(a, ot)
            c = Pow(a, -ot)
            return sqrt(ot) * (b * besseli(-ot, tt * a) + z * c * besseli(ot, tt * a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = S.One / (root(3, 6) * gamma(Rational(2, 3)))
        pf2 = z * root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(2, 3)], z ** 3 / 9) + pf2 * hyper([], [Rational(4, 3)], z ** 3 / 9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols
        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild('c', exclude=[z])
            d = Wild('d', exclude=[z])
            m = Wild('m', exclude=[z])
            n = Wild('n', exclude=[z])
            M = arg.match(c * (d * z ** n) ** m)
            if M is not None:
                m = M[m]
                if (3 * m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d * z ** n) ** m / (d ** m * z ** (m * n))
                    newarg = c * d ** m * z ** (m * n)
                    return S.Half * (sqrt(3) * (S.One - pf) * airyai(newarg) + (S.One + pf) * airybi(newarg))