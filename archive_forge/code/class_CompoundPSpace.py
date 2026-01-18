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
class CompoundPSpace(PSpace):
    """
    A temporary Probability Space for the Compound Distribution. After
    Marginalization, this returns the corresponding Probability Space of the
    parent distribution.
    """

    def __new__(cls, s, distribution):
        s = _symbol_converter(s)
        if isinstance(distribution, ContinuousDistribution):
            return SingleContinuousPSpace(s, distribution)
        if isinstance(distribution, DiscreteDistribution):
            return SingleDiscretePSpace(s, distribution)
        if isinstance(distribution, SingleFiniteDistribution):
            return SingleFinitePSpace(s, distribution)
        if not isinstance(distribution, CompoundDistribution):
            raise ValueError('%s should be an isinstance of CompoundDistribution' % distribution)
        return Basic.__new__(cls, s, distribution)

    @property
    def value(self):
        return RandomSymbol(self.symbol, self)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def is_Continuous(self):
        return self.distribution.is_Continuous

    @property
    def is_Finite(self):
        return self.distribution.is_Finite

    @property
    def is_Discrete(self):
        return self.distribution.is_Discrete

    @property
    def distribution(self):
        return self.args[1]

    @property
    def pdf(self):
        return self.distribution.pdf(self.symbol)

    @property
    def set(self):
        return self.distribution.set

    @property
    def domain(self):
        return self._get_newpspace().domain

    def _get_newpspace(self, evaluate=False):
        x = Dummy('x')
        parent_dist = self.distribution.args[0]
        func = Lambda(x, self.distribution.pdf(x, evaluate))
        new_pspace = self._transform_pspace(self.symbol, parent_dist, func)
        if new_pspace is not None:
            return new_pspace
        message = 'Compound Distribution for %s is not implemented yet' % str(parent_dist)
        raise NotImplementedError(message)

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

    def compute_density(self, expr, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        expr = expr.subs({self.value: new_pspace.value})
        return new_pspace.compute_density(expr, **kwargs)

    def compute_cdf(self, expr, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        expr = expr.subs({self.value: new_pspace.value})
        return new_pspace.compute_cdf(expr, **kwargs)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        new_pspace = self._get_newpspace(evaluate)
        expr = expr.subs({self.value: new_pspace.value})
        if rvs:
            rvs = rvs.subs({self.value: new_pspace.value})
        if isinstance(new_pspace, SingleFinitePSpace):
            return new_pspace.compute_expectation(expr, rvs, **kwargs)
        return new_pspace.compute_expectation(expr, rvs, evaluate, **kwargs)

    def probability(self, condition, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        condition = condition.subs({self.value: new_pspace.value})
        return new_pspace.probability(condition)

    def conditional_space(self, condition, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        condition = condition.subs({self.value: new_pspace.value})
        return new_pspace.conditional_space(condition)