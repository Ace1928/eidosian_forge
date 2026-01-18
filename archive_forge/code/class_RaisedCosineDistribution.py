from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random
class RaisedCosineDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    @property
    def set(self):
        return Interval(self.mu - self.s, self.mu + self.s)

    @staticmethod
    def check(mu, s):
        _value_check(s > 0, 's must be positive')

    def pdf(self, x):
        mu, s = (self.mu, self.s)
        return Piecewise(((1 + cos(pi * (x - mu) / s)) / (2 * s), And(mu - s <= x, x <= mu + s)), (S.Zero, True))

    def _characteristic_function(self, t):
        mu, s = (self.mu, self.s)
        return Piecewise((exp(-I * pi * mu / s) / 2, Eq(t, -pi / s)), (exp(I * pi * mu / s) / 2, Eq(t, pi / s)), (pi ** 2 * sin(s * t) * exp(I * mu * t) / (s * t * (pi ** 2 - s ** 2 * t ** 2)), True))

    def _moment_generating_function(self, t):
        mu, s = (self.mu, self.s)
        return pi ** 2 * sinh(s * t) * exp(mu * t) / (s * t * (pi ** 2 + s ** 2 * t ** 2))