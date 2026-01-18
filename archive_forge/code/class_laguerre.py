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
class laguerre(OrthogonalPolynomial):
    """
    Returns the $n$th Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import laguerre, diff
    >>> from sympy.abc import x, n
    >>> laguerre(0, x)
    1
    >>> laguerre(1, x)
    1 - x
    >>> laguerre(2, x)
    x**2/2 - 2*x + 1
    >>> laguerre(3, x)
    -x**3/6 + 3*x**2/2 - 3*x + 1

    >>> laguerre(n, x)
    laguerre(n, x)

    >>> diff(laguerre(n, x), x)
    -assoc_laguerre(n - 1, 1, x)

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \\ge 0`.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    assoc_laguerre,
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

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial
    .. [2] https://mathworld.wolfram.com/LaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """
    _ortho_poly = staticmethod(laguerre_poly)

    @classmethod
    def eval(cls, n, x):
        if n.is_integer is False:
            raise ValueError('Error: n should be an integer.')
        if not n.is_Number:
            if n.could_extract_minus_sign() and (not (-n - 1).could_extract_minus_sign()):
                return exp(x) * laguerre(-n - 1, -x)
            if x.is_zero:
                return S.One
            elif x is S.NegativeInfinity:
                return S.Infinity
            elif x is S.Infinity:
                return S.NegativeOne ** n * S.Infinity
        elif n.is_negative:
            return exp(x) * laguerre(-n - 1, -x)
        else:
            return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            n, x = self.args
            return -assoc_laguerre(n - 1, 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        if n.is_negative:
            return exp(x) * self._eval_rewrite_as_Sum(-n - 1, -x, **kwargs)
        if n.is_integer is False:
            raise ValueError('Error: n should be an integer.')
        k = Dummy('k')
        kern = RisingFactorial(-n, k) / factorial(k) ** 2 * x ** k
        return Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        return self._eval_rewrite_as_Sum(n, x, **kwargs)