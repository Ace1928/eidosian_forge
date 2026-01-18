from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions import assoc_legendre
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
def Ynm_c(n, m, theta, phi):
    """
    Conjugate spherical harmonics defined as

    .. math::
        \\overline{Y_n^m(\\theta, \\varphi)} := (-1)^m Y_n^{-m}(\\theta, \\varphi).

    Examples
    ========

    >>> from sympy import Ynm_c, Symbol, simplify
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")
    >>> Ynm_c(n, m, theta, phi)
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)
    >>> Ynm_c(n, m, -theta, phi)
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    For specific integers $n$ and $m$ we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Ynm_c(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))
    >>> simplify(Ynm_c(1, -1, theta, phi).expand(func=True))
    sqrt(6)*exp(I*(-phi + 2*conjugate(phi)))*sin(theta)/(4*sqrt(pi))

    See Also
    ========

    Ynm, Znm

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/

    """
    return conjugate(Ynm(n, m, theta, phi))