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
def _get_newpspace(self, evaluate=False):
    x = Dummy('x')
    parent_dist = self.distribution.args[0]
    func = Lambda(x, self.distribution.pdf(x, evaluate))
    new_pspace = self._transform_pspace(self.symbol, parent_dist, func)
    if new_pspace is not None:
        return new_pspace
    message = 'Compound Distribution for %s is not implemented yet' % str(parent_dist)
    raise NotImplementedError(message)