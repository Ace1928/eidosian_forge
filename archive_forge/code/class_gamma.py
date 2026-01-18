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
class gamma(Function):
    """
    The gamma function

    .. math::
        \\Gamma(x) := \\int^{\\infty}_{0} t^{x-1} e^{-t} \\mathrm{d}t.

    Explanation
    ===========

    The ``gamma`` function implements the function which passes through the
    values of the factorial function (i.e., $\\Gamma(n) = (n - 1)!$ when n is
    an integer). More generally, $\\Gamma(z)$ is defined in the whole complex
    plane except at the negative integers where there are simple poles.

    Examples
    ========

    >>> from sympy import S, I, pi, gamma
    >>> from sympy.abc import x

    Several special values are known:

    >>> gamma(1)
    1
    >>> gamma(4)
    6
    >>> gamma(S(3)/2)
    sqrt(pi)/2

    The ``gamma`` function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(gamma(x))
    gamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(gamma(x), x)
    gamma(x)*polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(gamma(x), x, 0, 3)
    1/x - EulerGamma + x*(EulerGamma**2/2 + pi**2/12) + x**2*(-EulerGamma*pi**2/12 - zeta(3)/3 - EulerGamma**3/6) + O(x**3)

    We can numerically evaluate the ``gamma`` function to arbitrary precision
    on the whole complex plane:

    >>> gamma(pi).evalf(40)
    2.288037795340032417959588909060233922890
    >>> gamma(1+I).evalf(20)
    0.49801566811835604271 - 0.15494982830181068512*I

    See Also
    ========

    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/GammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma/

    """
    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.func(self.args[0]) * polygamma(0, self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is oo:
                return oo
            elif intlike(arg):
                if arg.is_positive:
                    return factorial(arg - 1)
                else:
                    return S.ComplexInfinity
            elif arg.is_Rational:
                if arg.q == 2:
                    n = abs(arg.p) // arg.q
                    if arg.is_positive:
                        k, coeff = (n, S.One)
                    else:
                        n = k = n + 1
                        if n & 1 == 0:
                            coeff = S.One
                        else:
                            coeff = S.NegativeOne
                    coeff *= prod(range(3, 2 * k, 2))
                    if arg.is_positive:
                        return coeff * sqrt(pi) / 2 ** n
                    else:
                        return 2 ** n * sqrt(pi) / coeff

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        if arg.is_Rational:
            if abs(arg.p) > arg.q:
                x = Dummy('x')
                n = arg.p // arg.q
                p = arg.p - n * arg.q
                return self.func(x + n)._eval_expand_func().subs(x, Rational(p, arg.q))
        if arg.is_Add:
            coeff, tail = arg.as_coeff_add()
            if coeff and coeff.q != 1:
                intpart = floor(coeff)
                tail = (coeff - intpart,) + tail
                coeff = intpart
            tail = arg._new_rawargs(*tail, reeval=False)
            return self.func(tail) * RisingFactorial(tail, coeff)
        return self.func(*self.args)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        x = self.args[0]
        if x.is_nonpositive and x.is_integer:
            return False
        if intlike(x) and x <= 0:
            return False
        if x.is_positive or x.is_noninteger:
            return True

    def _eval_is_positive(self):
        x = self.args[0]
        if x.is_positive:
            return True
        elif x.is_noninteger:
            return floor(x).is_even

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(loggamma(z))

    def _eval_rewrite_as_factorial(self, z, **kwargs):
        return factorial(z - 1)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if not (x0.is_Integer and x0 <= 0):
            return super()._eval_nseries(x, n, logx)
        t = self.args[0] - x0
        return (self.func(t + 1) / rf(self.args[0], -x0 + 1))._eval_nseries(x, n, logx)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0]
        x0 = arg.subs(x, 0)
        if x0.is_integer and x0.is_nonpositive:
            n = -x0
            res = S.NegativeOne ** n / self.func(n + 1)
            return res / (arg + n).as_leading_term(x)
        elif not x0.is_infinite:
            return self.func(x0)
        raise PoleError()