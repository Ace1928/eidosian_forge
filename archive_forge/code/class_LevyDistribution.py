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
class LevyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'c')

    @property
    def set(self):
        return Interval(self.mu, oo)

    @staticmethod
    def check(mu, c):
        _value_check(c > 0, 'c (scale parameter) must be positive')
        _value_check(mu.is_real, 'mu (location parameter) must be real')

    def pdf(self, x):
        mu, c = (self.mu, self.c)
        return sqrt(c / (2 * pi)) * exp(-c / (2 * (x - mu))) / (x - mu) ** (S.One + S.Half)

    def _cdf(self, x):
        mu, c = (self.mu, self.c)
        return erfc(sqrt(c / (2 * (x - mu))))

    def _characteristic_function(self, t):
        mu, c = (self.mu, self.c)
        return exp(I * mu * t - sqrt(-2 * I * c * t))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function of Levy distribution does not exist.')