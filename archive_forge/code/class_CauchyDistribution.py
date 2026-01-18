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
class CauchyDistribution(SingleContinuousDistribution):
    _argnames = ('x0', 'gamma')

    @staticmethod
    def check(x0, gamma):
        _value_check(gamma > 0, 'Scale parameter Gamma must be positive.')
        _value_check(x0.is_real, 'Location parameter must be real.')

    def pdf(self, x):
        return 1 / (pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))

    def _cdf(self, x):
        x0, gamma = (self.x0, self.gamma)
        return 1 / pi * atan((x - x0) / gamma) + S.Half

    def _characteristic_function(self, t):
        return exp(self.x0 * I * t - self.gamma * Abs(t))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the Cauchy distribution does not exist.')

    def _quantile(self, p):
        return self.x0 + self.gamma * tan(pi * (p - S.Half))