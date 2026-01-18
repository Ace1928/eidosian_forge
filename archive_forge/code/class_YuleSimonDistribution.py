from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import (polylog, zeta)
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import _value_check, is_random
class YuleSimonDistribution(SingleDiscreteDistribution):
    _argnames = ('rho',)
    set = S.Naturals

    @staticmethod
    def check(rho):
        _value_check(rho > 0, 'rho should be positive')

    def pdf(self, k):
        rho = self.rho
        return rho * beta(k, rho + 1)

    def _cdf(self, x):
        return Piecewise((1 - floor(x) * beta(floor(x), self.rho + 1), x >= 1), (0, True))

    def _characteristic_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(I * t)) * exp(I * t) / (rho + 1)

    def _moment_generating_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(t)) * exp(t) / (rho + 1)