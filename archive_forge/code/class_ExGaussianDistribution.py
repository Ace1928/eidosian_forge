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
class ExGaussianDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std', 'rate')
    set = Interval(-oo, oo)

    @staticmethod
    def check(mean, std, rate):
        _value_check(std > 0, 'Standard deviation of ExGaussian must be positive.')
        _value_check(rate > 0, 'Rate of ExGaussian must be positive.')

    def pdf(self, x):
        mean, std, rate = (self.mean, self.std, self.rate)
        term1 = rate / 2
        term2 = exp(rate * (2 * mean + rate * std ** 2 - 2 * x) / 2)
        term3 = erfc((mean + rate * std ** 2 - x) / (sqrt(2) * std))
        return term1 * term2 * term3

    def _cdf(self, x):
        from sympy.stats import cdf
        mean, std, rate = (self.mean, self.std, self.rate)
        u = rate * (x - mean)
        v = rate * std
        GaussianCDF1 = cdf(Normal('x', 0, v))(u)
        GaussianCDF2 = cdf(Normal('x', v ** 2, v))(u)
        return GaussianCDF1 - exp(-u + v ** 2 / 2 + log(GaussianCDF2))

    def _characteristic_function(self, t):
        mean, std, rate = (self.mean, self.std, self.rate)
        term1 = (1 - I * t / rate) ** (-1)
        term2 = exp(I * mean * t - std ** 2 * t ** 2 / 2)
        return term1 * term2

    def _moment_generating_function(self, t):
        mean, std, rate = (self.mean, self.std, self.rate)
        term1 = (1 - t / rate) ** (-1)
        term2 = exp(mean * t + std ** 2 * t ** 2 / 2)
        return term1 * term2