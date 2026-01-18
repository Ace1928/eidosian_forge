from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.symbol import Dummy
from sympy.integrals.integrals import Integral
from sympy.stats.rv import (NamedArgsMixin, random_symbols, _symbol_converter,
from sympy.stats.crv import ContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import DiscreteDistribution, SingleDiscretePSpace
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace
from sympy.stats.crv_types import ContinuousDistributionHandmade
from sympy.stats.drv_types import DiscreteDistributionHandmade
from sympy.stats.frv_types import FiniteDistributionHandmade
def _transform_pspace(self, sym, dist, pdf):
    """
        This function returns the new pspace of the distribution using handmade
        Distributions and their corresponding pspace.
        """
    pdf = Lambda(sym, pdf(sym))
    _set = dist.set
    if isinstance(dist, ContinuousDistribution):
        return SingleContinuousPSpace(sym, ContinuousDistributionHandmade(pdf, _set))
    elif isinstance(dist, DiscreteDistribution):
        return SingleDiscretePSpace(sym, DiscreteDistributionHandmade(pdf, _set))
    elif isinstance(dist, SingleFiniteDistribution):
        dens = {k: pdf(k) for k in _set}
        return SingleFinitePSpace(sym, FiniteDistributionHandmade(dens))