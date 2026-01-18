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
class jacobi(OrthogonalPolynomial):
    """
    Jacobi polynomial $P_n^{\\left(\\alpha, \\beta\\right)}(x)$.

    Explanation
    ===========

    ``jacobi(n, alpha, beta, x)`` gives the $n$th Jacobi polynomial
    in $x$, $P_n^{\\left(\\alpha, \\beta\\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\\left(1-x\\right)^\\alpha \\left(1+x\\right)^\\beta$.

    Examples
    ========

    >>> from sympy import jacobi, S, conjugate, diff
    >>> from sympy.abc import a, b, n, x

    >>> jacobi(0, a, b, x)
    1
    >>> jacobi(1, a, b, x)
    a/2 - b/2 + x*(a/2 + b/2 + 1)
    >>> jacobi(2, a, b, x)
    a**2/8 - a*b/4 - a/8 + b**2/8 - b/8 + x**2*(a**2/8 + a*b/4 + 7*a/8 + b**2/8 + 7*b/8 + 3/2) + x*(a**2/4 + 3*a/4 - b**2/4 - 3*b/4) - 1/2

    >>> jacobi(n, a, b, x)
    jacobi(n, a, b, x)

    >>> jacobi(n, a, a, x)
    RisingFactorial(a + 1, n)*gegenbauer(n,
        a + 1/2, x)/RisingFactorial(2*a + 1, n)

    >>> jacobi(n, 0, 0, x)
    legendre(n, x)

    >>> jacobi(n, S(1)/2, S(1)/2, x)
    RisingFactorial(3/2, n)*chebyshevu(n, x)/factorial(n + 1)

    >>> jacobi(n, -S(1)/2, -S(1)/2, x)
    RisingFactorial(1/2, n)*chebyshevt(n, x)/factorial(n)

    >>> jacobi(n, a, b, -x)
    (-1)**n*jacobi(n, b, a, x)

    >>> jacobi(n, a, b, 0)
    gamma(a + n + 1)*hyper((-b - n, -n), (a + 1,), -1)/(2**n*factorial(n)*gamma(a + 1))
    >>> jacobi(n, a, b, 1)
    RisingFactorial(a + 1, n)/factorial(n)

    >>> conjugate(jacobi(n, a, b, x))
    jacobi(n, conjugate(a), conjugate(b), conjugate(x))

    >>> diff(jacobi(n,a,b,x), x)
    (a/2 + b/2 + n/2 + 1/2)*jacobi(n - 1, a + 1, b + 1, x)

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """

    @classmethod
    def eval(cls, n, a, b, x):
        if a == b:
            if a == Rational(-1, 2):
                return RisingFactorial(S.Half, n) / factorial(n) * chebyshevt(n, x)
            elif a.is_zero:
                return legendre(n, x)
            elif a == S.Half:
                return RisingFactorial(3 * S.Half, n) / factorial(n + 1) * chebyshevu(n, x)
            else:
                return RisingFactorial(a + 1, n) / RisingFactorial(2 * a + 1, n) * gegenbauer(n, a + S.Half, x)
        elif b == -a:
            return gamma(n + a + 1) / gamma(n + 1) * (1 + x) ** (a / 2) / (1 - x) ** (a / 2) * assoc_legendre(n, -a, x)
        if not n.is_Number:
            if x.could_extract_minus_sign():
                return S.NegativeOne ** n * jacobi(n, b, a, -x)
            if x.is_zero:
                return 2 ** (-n) * gamma(a + n + 1) / (gamma(a + 1) * factorial(n)) * hyper([-b - n, -n], [a + 1], -1)
            if x == S.One:
                return RisingFactorial(a + 1, n) / factorial(n)
            elif x is S.Infinity:
                if n.is_positive:
                    if (a + b + 2 * n).is_integer:
                        raise ValueError('Error. a + b + 2*n should not be an integer.')
                    return RisingFactorial(a + b + n + 1, n) * S.Infinity
        else:
            return jacobi_poly(n, a, b, x)

    def fdiff(self, argindex=4):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            n, a, b, x = self.args
            k = Dummy('k')
            f1 = 1 / (a + b + n + k + 1)
            f2 = (a + b + 2 * k + 1) * RisingFactorial(b + k + 1, n - k) / ((n - k) * RisingFactorial(a + b + k + 1, n - k))
            return Sum(f1 * (jacobi(n, a, b, x) + f2 * jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 3:
            n, a, b, x = self.args
            k = Dummy('k')
            f1 = 1 / (a + b + n + k + 1)
            f2 = (-1) ** (n - k) * ((a + b + 2 * k + 1) * RisingFactorial(a + k + 1, n - k) / ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            return Sum(f1 * (jacobi(n, a, b, x) + f2 * jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 4:
            n, a, b, x = self.args
            return S.Half * (a + b + n + 1) * jacobi(n - 1, a + 1, b + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, b, x, **kwargs):
        from sympy.concrete.summations import Sum
        if n.is_negative or n.is_integer is False:
            raise ValueError('Error: n should be a non-negative integer.')
        k = Dummy('k')
        kern = RisingFactorial(-n, k) * RisingFactorial(a + b + n + 1, k) * RisingFactorial(a + k + 1, n - k) / factorial(k) * ((1 - x) / 2) ** k
        return 1 / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, a, b, x, **kwargs):
        return self._eval_rewrite_as_Sum(n, a, b, x, **kwargs)

    def _eval_conjugate(self):
        n, a, b, x = self.args
        return self.func(n, a.conjugate(), b.conjugate(), x.conjugate())