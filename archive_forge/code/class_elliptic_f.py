from sympy.core import S, pi, I, Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, tan
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
class elliptic_f(Function):
    """
    The Legendre incomplete elliptic integral of the first
    kind, defined by

    .. math:: F\\left(z\\middle| m\\right) =
              \\int_0^z \\frac{dt}{\\sqrt{1 - m \\sin^2 t}}

    Explanation
    ===========

    This function reduces to a complete elliptic integral of
    the first kind, $K(m)$, when $z = \\pi/2$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_f, I
    >>> from sympy.abc import z, m
    >>> elliptic_f(z, m).series(z)
    z + z**5*(3*m**2/40 - m/30) + m*z**3/6 + O(z**6)
    >>> elliptic_f(3.0 + I/2, 1.0 + I)
    2.909449841483 + 1.74720545502474*I

    See Also
    ========

    elliptic_k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticF

    """

    @classmethod
    def eval(cls, z, m):
        if z.is_zero:
            return S.Zero
        if m.is_zero:
            return z
        k = 2 * z / pi
        if k.is_integer:
            return k * elliptic_k(m)
        elif m in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        elif z.could_extract_minus_sign():
            return -elliptic_f(-z, m)

    def fdiff(self, argindex=1):
        z, m = self.args
        fm = sqrt(1 - m * sin(z) ** 2)
        if argindex == 1:
            return 1 / fm
        elif argindex == 2:
            return elliptic_e(z, m) / (2 * m * (1 - m)) - elliptic_f(z, m) / (2 * m) - sin(2 * z) / (4 * (1 - m) * fm)
        raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        z, m = self.args
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(z.conjugate(), m.conjugate())

    def _eval_rewrite_as_Integral(self, *args):
        from sympy.integrals.integrals import Integral
        t = Dummy('t')
        z, m = (self.args[0], self.args[1])
        return Integral(1 / sqrt(1 - m * sin(t) ** 2), (t, 0, z))

    def _eval_is_zero(self):
        z, m = self.args
        if z.is_zero:
            return True
        if m.is_extended_real and m.is_infinite:
            return True