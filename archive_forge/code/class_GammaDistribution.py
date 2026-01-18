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
class GammaDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'theta')
    set = Interval(0, oo)

    @staticmethod
    def check(k, theta):
        _value_check(k > 0, 'k must be positive')
        _value_check(theta > 0, 'Theta must be positive')

    def pdf(self, x):
        k, theta = (self.k, self.theta)
        return x ** (k - 1) * exp(-x / theta) / (gamma(k) * theta ** k)

    def _cdf(self, x):
        k, theta = (self.k, self.theta)
        return Piecewise((lowergamma(k, S(x) / theta) / gamma(k), x > 0), (S.Zero, True))

    def _characteristic_function(self, t):
        return (1 - self.theta * I * t) ** (-self.k)

    def _moment_generating_function(self, t):
        return (1 - self.theta * t) ** (-self.k)