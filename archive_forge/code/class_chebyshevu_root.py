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
class chebyshevu_root(Function):
    """
    ``chebyshevu_root(n, k)`` returns the $k$th root (indexed from zero) of the
    $n$th Chebyshev polynomial of the second kind; that is, if $0 \\le k < n$,
    ``chebyshevu(n, chebyshevu_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevu, chebyshevu_root
    >>> chebyshevu_root(3, 2)
    -sqrt(2)/2
    >>> chebyshevu(3, chebyshevu_root(3, 2))
    0

    See Also
    ========

    chebyshevt, chebyshevt_root, chebyshevu,
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
    """

    @classmethod
    def eval(cls, n, k):
        if not (0 <= k and k < n):
            raise ValueError('must have 0 <= k < n, got k = %s and n = %s' % (k, n))
        return cos(S.Pi * (k + 1) / (n + 1))