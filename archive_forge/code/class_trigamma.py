from math import prod
from sympy.core import Add, S, Dummy, expand_func
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and, fuzzy_not
from sympy.core.numbers import Rational, pi, oo, I
from sympy.core.power import Pow
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.combinatorial.factorials import factorial, rf, RisingFactorial
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
class trigamma(Function):
    """
    The ``trigamma`` function is the second derivative of the ``loggamma``
    function

    .. math::
        \\psi^{(1)}(z) := \\frac{\\mathrm{d}^{2}}{\\mathrm{d} z^{2}} \\log\\Gamma(z).

    In this case, ``trigamma(z) = polygamma(1, z)``.

    Examples
    ========

    >>> from sympy import trigamma
    >>> trigamma(0)
    zoo
    >>> from sympy import Symbol
    >>> z = Symbol('z')
    >>> trigamma(z)
    polygamma(1, z)

    To retain ``trigamma`` as it is:

    >>> trigamma(0, evaluate=False)
    trigamma(0)
    >>> trigamma(z, evaluate=False)
    trigamma(z)


    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigamma_function
    .. [2] https://mathworld.wolfram.com/TrigammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/

    """

    def _eval_evalf(self, prec):
        z = self.args[0]
        nprec = prec_to_dps(prec)
        return polygamma(1, z).evalf(n=nprec)

    def fdiff(self, argindex=1):
        z = self.args[0]
        return polygamma(1, z).fdiff()

    def _eval_is_real(self):
        z = self.args[0]
        return polygamma(1, z).is_real

    def _eval_is_positive(self):
        z = self.args[0]
        return polygamma(1, z).is_positive

    def _eval_is_negative(self):
        z = self.args[0]
        return polygamma(1, z).is_negative

    def _eval_aseries(self, n, args0, x, logx):
        as_polygamma = self.rewrite(polygamma)
        args0 = [S.One] + args0
        return as_polygamma._eval_aseries(n, args0, x, logx)

    @classmethod
    def eval(cls, z):
        return polygamma(1, z)

    def _eval_expand_func(self, **hints):
        z = self.args[0]
        return polygamma(1, z).expand(func=True)

    def _eval_rewrite_as_zeta(self, z, **kwargs):
        return zeta(2, z)

    def _eval_rewrite_as_polygamma(self, z, **kwargs):
        return polygamma(1, z)

    def _eval_rewrite_as_harmonic(self, z, **kwargs):
        return -harmonic(z - 1, 2) + pi ** 2 / 6

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        z = self.args[0]
        return polygamma(1, z).as_leading_term(x)