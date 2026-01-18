from sympy.core import S, pi, I, Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, tan
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
class elliptic_k(Function):
    """
    The complete elliptic integral of the first kind, defined by

    .. math:: K(m) = F\\left(\\tfrac{\\pi}{2}\\middle| m\\right)

    where $F\\left(z\\middle| m\\right)$ is the Legendre incomplete
    elliptic integral of the first kind.

    Explanation
    ===========

    The function $K(m)$ is a single-valued function on the complex
    plane with branch cut along the interval $(1, \\infty)$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_k, I
    >>> from sympy.abc import m
    >>> elliptic_k(0)
    pi/2
    >>> elliptic_k(1.0 + I)
    1.50923695405127 + 0.625146415202697*I
    >>> elliptic_k(m).series(n=3)
    pi/2 + pi*m/8 + 9*pi*m**2/128 + O(m**3)

    See Also
    ========

    elliptic_f

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticK

    """

    @classmethod
    def eval(cls, m):
        if m.is_zero:
            return pi * S.Half
        elif m is S.Half:
            return 8 * pi ** Rational(3, 2) / gamma(Rational(-1, 4)) ** 2
        elif m is S.One:
            return S.ComplexInfinity
        elif m is S.NegativeOne:
            return gamma(Rational(1, 4)) ** 2 / (4 * sqrt(2 * pi))
        elif m in (S.Infinity, S.NegativeInfinity, I * S.Infinity, I * S.NegativeInfinity, S.ComplexInfinity):
            return S.Zero

    def fdiff(self, argindex=1):
        m = self.args[0]
        return (elliptic_e(m) - (1 - m) * elliptic_k(m)) / (2 * m * (1 - m))

    def _eval_conjugate(self):
        m = self.args[0]
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.simplify import hyperexpand
        return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))

    def _eval_rewrite_as_hyper(self, m, **kwargs):
        return pi * S.Half * hyper((S.Half, S.Half), (S.One,), m)

    def _eval_rewrite_as_meijerg(self, m, **kwargs):
        return meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -m) / 2

    def _eval_is_zero(self):
        m = self.args[0]
        if m.is_infinite:
            return True

    def _eval_rewrite_as_Integral(self, *args):
        from sympy.integrals.integrals import Integral
        t = Dummy('t')
        m = self.args[0]
        return Integral(1 / sqrt(1 - m * sin(t) ** 2), (t, 0, pi / 2))