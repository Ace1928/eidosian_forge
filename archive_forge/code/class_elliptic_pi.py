from sympy.core import S, pi, I, Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, tan
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
class elliptic_pi(Function):
    """
    Called with three arguments $n$, $z$ and $m$, evaluates the
    Legendre incomplete elliptic integral of the third kind, defined by

    .. math:: \\Pi\\left(n; z\\middle| m\\right) = \\int_0^z \\frac{dt}
              {\\left(1 - n \\sin^2 t\\right) \\sqrt{1 - m \\sin^2 t}}

    Called with two arguments $n$ and $m$, evaluates the complete
    elliptic integral of the third kind:

    .. math:: \\Pi\\left(n\\middle| m\\right) =
              \\Pi\\left(n; \\tfrac{\\pi}{2}\\middle| m\\right)

    Explanation
    ===========

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_pi, I
    >>> from sympy.abc import z, n, m
    >>> elliptic_pi(n, z, m).series(z, n=4)
    z + z**3*(m/6 + n/3) + O(z**4)
    >>> elliptic_pi(0.5 + I, 1.0 - I, 1.2)
    2.50232379629182 - 0.760939574180767*I
    >>> elliptic_pi(0, 0)
    pi/2
    >>> elliptic_pi(1.0 - I/3, 2.0 + I)
    3.29136443417283 + 0.32555634906645*I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticPi3
    .. [3] https://functions.wolfram.com/EllipticIntegrals/EllipticPi

    """

    @classmethod
    def eval(cls, n, m, z=None):
        if z is not None:
            n, z, m = (n, m, z)
            if n.is_zero:
                return elliptic_f(z, m)
            elif n is S.One:
                return elliptic_f(z, m) + (sqrt(1 - m * sin(z) ** 2) * tan(z) - elliptic_e(z, m)) / (1 - m)
            k = 2 * z / pi
            if k.is_integer:
                return k * elliptic_pi(n, m)
            elif m.is_zero:
                return atanh(sqrt(n - 1) * tan(z)) / sqrt(n - 1)
            elif n == m:
                return elliptic_f(z, n) - elliptic_pi(1, z, n) + tan(z) / sqrt(1 - n * sin(z) ** 2)
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif z.could_extract_minus_sign():
                return -elliptic_pi(n, -z, m)
            if n.is_zero:
                return elliptic_f(z, m)
            if m.is_extended_real and m.is_infinite or (n.is_extended_real and n.is_infinite):
                return S.Zero
        else:
            if n.is_zero:
                return elliptic_k(m)
            elif n is S.One:
                return S.ComplexInfinity
            elif m.is_zero:
                return pi / (2 * sqrt(1 - n))
            elif m == S.One:
                return S.NegativeInfinity / sign(n - 1)
            elif n == m:
                return elliptic_e(n) / (1 - n)
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            if n.is_zero:
                return elliptic_k(m)
            if m.is_extended_real and m.is_infinite or (n.is_extended_real and n.is_infinite):
                return S.Zero

    def _eval_conjugate(self):
        if len(self.args) == 3:
            n, z, m = self.args
            if (n.is_real and (n - 1).is_positive) is False and (m.is_real and (m - 1).is_positive) is False:
                return self.func(n.conjugate(), z.conjugate(), m.conjugate())
        else:
            n, m = self.args
            return self.func(n.conjugate(), m.conjugate())

    def fdiff(self, argindex=1):
        if len(self.args) == 3:
            n, z, m = self.args
            fm, fn = (sqrt(1 - m * sin(z) ** 2), 1 - n * sin(z) ** 2)
            if argindex == 1:
                return (elliptic_e(z, m) + (m - n) * elliptic_f(z, m) / n + (n ** 2 - m) * elliptic_pi(n, z, m) / n - n * fm * sin(2 * z) / (2 * fn)) / (2 * (m - n) * (n - 1))
            elif argindex == 2:
                return 1 / (fm * fn)
            elif argindex == 3:
                return (elliptic_e(z, m) / (m - 1) + elliptic_pi(n, z, m) - m * sin(2 * z) / (2 * (m - 1) * fm)) / (2 * (n - m))
        else:
            n, m = self.args
            if argindex == 1:
                return (elliptic_e(m) + (m - n) * elliptic_k(m) / n + (n ** 2 - m) * elliptic_pi(n, m) / n) / (2 * (m - n) * (n - 1))
            elif argindex == 2:
                return (elliptic_e(m) / (m - 1) + elliptic_pi(n, m)) / (2 * (n - m))
        raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, *args):
        from sympy.integrals.integrals import Integral
        if len(self.args) == 2:
            n, m, z = (self.args[0], self.args[1], pi / 2)
        else:
            n, z, m = self.args
        t = Dummy('t')
        return Integral(1 / ((1 - n * sin(t) ** 2) * sqrt(1 - m * sin(t) ** 2)), (t, 0, z))