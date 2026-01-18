from sympy.core import S, pi, I, Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, tan
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg

    Called with three arguments $n$, $z$ and $m$, evaluates the
    Legendre incomplete elliptic integral of the third kind, defined by

    .. math:: \Pi\left(n; z\middle| m\right) = \int_0^z \frac{dt}
              {\left(1 - n \sin^2 t\right) \sqrt{1 - m \sin^2 t}}

    Called with two arguments $n$ and $m$, evaluates the complete
    elliptic integral of the third kind:

    .. math:: \Pi\left(n\middle| m\right) =
              \Pi\left(n; \tfrac{\pi}{2}\middle| m\right)

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

    