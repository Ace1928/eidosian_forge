from functools import singledispatch
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import exp
from sympy.utilities.lambdify import lambdify
from sympy.external import import_module
from sympy.stats import DiscreteDistributionHandmade
from sympy.stats.crv import SingleContinuousDistribution
from sympy.stats.crv_types import ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
from sympy.stats.drv_types import GeometricDistribution, LogarithmicDistribution, NegativeBinomialDistribution, \
from sympy.stats.frv import SingleFiniteDistribution
class scipy_pdf(scipy.stats.rv_continuous):

    def _pdf(dist, x):
        return handmade_pdf(x)