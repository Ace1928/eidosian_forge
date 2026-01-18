from functools import wraps
from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn
from mpmath import mp, workprec
class hankel2(BesselBase):
    """
    Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\\nu^{(2)} = J_\\nu(z) - iY_\\nu(z),

    where $J_\\nu(z)$ is the Bessel function of the first kind, and
    $Y_\\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation, and linearly independent from
    $H_\\nu^{(1)}$.

    Examples
    ========

    >>> from sympy import hankel2
    >>> from sympy.abc import z, n
    >>> hankel2(n, z).diff(z)
    hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2

    See Also
    ========

    hankel1, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH2/

    """
    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return hankel1(self.order.conjugate(), z.conjugate())