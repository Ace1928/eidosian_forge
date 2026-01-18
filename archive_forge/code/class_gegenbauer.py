from sympy.core import Rational
from sympy.core.function import Function, ArgumentIndexError
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial, factorial, RisingFactorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sec
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import (chebyshevt_poly, chebyshevu_poly,
class gegenbauer(OrthogonalPolynomial):
    """
    Gegenbauer polynomial $C_n^{\\left(\\alpha\\right)}(x)$.

    Explanation
    ===========

    ``gegenbauer(n, alpha, x)`` gives the $n$th Gegenbauer polynomial
    in $x$, $C_n^{\\left(\\alpha\\right)}(x)$.

    The Gegenbauer polynomials are orthogonal on $[-1, 1]$ with
    respect to the weight $\\left(1-x^2\\right)^{\\alpha-\\frac{1}{2}}$.

    Examples
    ========

    >>> from sympy import gegenbauer, conjugate, diff
    >>> from sympy.abc import n,a,x
    >>> gegenbauer(0, a, x)
    1
    >>> gegenbauer(1, a, x)
    2*a*x
    >>> gegenbauer(2, a, x)
    -a + x**2*(2*a**2 + 2*a)
    >>> gegenbauer(3, a, x)
    x**3*(4*a**3/3 + 4*a**2 + 8*a/3) + x*(-2*a**2 - 2*a)

    >>> gegenbauer(n, a, x)
    gegenbauer(n, a, x)
    >>> gegenbauer(n, a, -x)
    (-1)**n*gegenbauer(n, a, x)

    >>> gegenbauer(n, a, 0)
    2**n*sqrt(pi)*gamma(a + n/2)/(gamma(a)*gamma(1/2 - n/2)*gamma(n + 1))
    >>> gegenbauer(n, a, 1)
    gamma(2*a + n)/(gamma(2*a)*gamma(n + 1))

    >>> conjugate(gegenbauer(n, a, x))
    gegenbauer(n, conjugate(a), conjugate(x))

    >>> diff(gegenbauer(n, a, x), x)
    2*a*gegenbauer(n - 1, a + 1, x)

    See Also
    ========

    jacobi,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gegenbauer_polynomials
    .. [2] https://mathworld.wolfram.com/GegenbauerPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/GegenbauerC3/

    """

    @classmethod
    def eval(cls, n, a, x):
        if n.is_negative:
            return S.Zero
        if a == S.Half:
            return legendre(n, x)
        elif a == S.One:
            return chebyshevu(n, x)
        elif a == S.NegativeOne:
            return S.Zero
        if not n.is_Number:
            if x == S.NegativeOne:
                if (re(a) > S.Half) == True:
                    return S.ComplexInfinity
                else:
                    return cos(S.Pi * (a + n)) * sec(S.Pi * a) * gamma(2 * a + n) / (gamma(2 * a) * gamma(n + 1))
            if x.could_extract_minus_sign():
                return S.NegativeOne ** n * gegenbauer(n, a, -x)
            if x.is_zero:
                return 2 ** n * sqrt(S.Pi) * gamma(a + S.Half * n) / (gamma((1 - n) / 2) * gamma(n + 1) * gamma(a))
            if x == S.One:
                return gamma(2 * a + n) / (gamma(2 * a) * gamma(n + 1))
            elif x is S.Infinity:
                if n.is_positive:
                    return RisingFactorial(a, n) * S.Infinity
        else:
            return gegenbauer_poly(n, a, x)

    def fdiff(self, argindex=3):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            n, a, x = self.args
            k = Dummy('k')
            factor1 = 2 * (1 + (-1) ** (n - k)) * (k + a) / ((k + n + 2 * a) * (n - k))
            factor2 = 2 * (k + 1) / ((k + 2 * a) * (2 * k + 2 * a + 1)) + 2 / (k + n + 2 * a)
            kern = factor1 * gegenbauer(k, a, x) + factor2 * gegenbauer(n, a, x)
            return Sum(kern, (k, 0, n - 1))
        elif argindex == 3:
            n, a, x = self.args
            return 2 * a * gegenbauer(n - 1, a + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy('k')
        kern = (-1) ** k * RisingFactorial(a, n - k) * (2 * x) ** (n - 2 * k) / (factorial(k) * factorial(n - 2 * k))
        return Sum(kern, (k, 0, floor(n / 2)))

    def _eval_rewrite_as_polynomial(self, n, a, x, **kwargs):
        return self._eval_rewrite_as_Sum(n, a, x, **kwargs)

    def _eval_conjugate(self):
        n, a, x = self.args
        return self.func(n, a.conjugate(), x.conjugate())