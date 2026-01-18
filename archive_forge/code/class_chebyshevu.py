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
class chebyshevu(OrthogonalPolynomial):
    """
    Chebyshev polynomial of the second kind, $U_n(x)$.

    Explanation
    ===========

    ``chebyshevu(n, x)`` gives the $n$th Chebyshev polynomial of the second
    kind in x, $U_n(x)$.

    The Chebyshev polynomials of the second kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\\sqrt{1-x^2}$.

    Examples
    ========

    >>> from sympy import chebyshevu, diff
    >>> from sympy.abc import n,x
    >>> chebyshevu(0, x)
    1
    >>> chebyshevu(1, x)
    2*x
    >>> chebyshevu(2, x)
    4*x**2 - 1

    >>> chebyshevu(n, x)
    chebyshevu(n, x)
    >>> chebyshevu(n, -x)
    (-1)**n*chebyshevu(n, x)
    >>> chebyshevu(-n, x)
    -chebyshevu(n - 2, x)

    >>> chebyshevu(n, 0)
    cos(pi*n/2)
    >>> chebyshevu(n, 1)
    n + 1

    >>> diff(chebyshevu(n, x), x)
    (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu_root,
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

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """
    _ortho_poly = staticmethod(chebyshevu_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            if x.could_extract_minus_sign():
                return S.NegativeOne ** n * chebyshevu(n, -x)
            if n.could_extract_minus_sign():
                if n == S.NegativeOne:
                    return S.Zero
                elif not (-n - 2).could_extract_minus_sign():
                    return -chebyshevu(-n - 2, x)
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One + n
            elif x is S.Infinity:
                return S.Infinity
        elif n.is_negative:
            if n == S.NegativeOne:
                return S.Zero
            else:
                return -cls._eval_at_order(-n - 2, x)
        else:
            return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            n, x = self.args
            return ((n + 1) * chebyshevt(n + 1, x) - x * chebyshevu(n, x)) / (x ** 2 - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy('k')
        kern = S.NegativeOne ** k * factorial(n - k) * (2 * x) ** (n - 2 * k) / (factorial(k) * factorial(n - 2 * k))
        return Sum(kern, (k, 0, floor(n / 2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        return self._eval_rewrite_as_Sum(n, x, **kwargs)