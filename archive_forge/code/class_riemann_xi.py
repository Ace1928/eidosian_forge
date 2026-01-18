from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, Function
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly
class riemann_xi(Function):
    """
    Riemann Xi function.

    Examples
    ========

    The Riemann Xi function is closely related to the Riemann zeta function.
    The zeros of Riemann Xi function are precisely the non-trivial zeros
    of the zeta function.

    >>> from sympy import riemann_xi, zeta
    >>> from sympy.abc import s
    >>> riemann_xi(s).rewrite(zeta)
    s*(s - 1)*gamma(s/2)*zeta(s)/(2*pi**(s/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Riemann_Xi_function

    """

    @classmethod
    def eval(cls, s):
        from sympy.functions.special.gamma_functions import gamma
        z = zeta(s)
        if s in (S.Zero, S.One):
            return S.Half
        if not isinstance(z, zeta):
            return s * (s - 1) * gamma(s / 2) * z / (2 * pi ** (s / 2))

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return s * (s - 1) * gamma(s / 2) * zeta(s) / (2 * pi ** (s / 2))