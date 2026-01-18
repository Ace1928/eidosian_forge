from math import prod
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.sets.sets import ProductSet
from sympy.tensor.indexed import Indexed
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation
from sympy.core.containers import Tuple
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices import ImmutableMatrix, matrix2numpy, list2numpy
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import (ProductPSpace, NamedArgsMixin, Distribution,
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent
from sympy.external import import_module
class JointPSpace(ProductPSpace):
    """
    Represents a joint probability space. Represented using symbols for
    each component and a distribution.
    """

    def __new__(cls, sym, dist):
        if isinstance(dist, SingleContinuousDistribution):
            return SingleContinuousPSpace(sym, dist)
        if isinstance(dist, SingleDiscreteDistribution):
            return SingleDiscretePSpace(sym, dist)
        sym = _symbol_converter(sym)
        return Basic.__new__(cls, sym, dist)

    @property
    def set(self):
        return self.domain.set

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[1]

    @property
    def value(self):
        return JointRandomSymbol(self.symbol, self)

    @property
    def component_count(self):
        _set = self.distribution.set
        if isinstance(_set, ProductSet):
            return S(len(_set.args))
        elif isinstance(_set, Product):
            return _set.limits[0][-1]
        return S.One

    @property
    def pdf(self):
        sym = [Indexed(self.symbol, i) for i in range(self.component_count)]
        return self.distribution(*sym)

    @property
    def domain(self):
        rvs = random_symbols(self.distribution)
        if not rvs:
            return SingleDomain(self.symbol, self.distribution.set)
        return ProductDomain(*[rv.pspace.domain for rv in rvs])

    def component_domain(self, index):
        return self.set.args[index]

    def marginal_distribution(self, *indices):
        count = self.component_count
        if count.atoms(Symbol):
            raise ValueError('Marginal distributions cannot be computed for symbolic dimensions. It is a work under progress.')
        orig = [Indexed(self.symbol, i) for i in range(count)]
        all_syms = [Symbol(str(i)) for i in orig]
        replace_dict = dict(zip(all_syms, orig))
        sym = tuple((Symbol(str(Indexed(self.symbol, i))) for i in indices))
        limits = [[i] for i in all_syms if i not in sym]
        index = 0
        for i in range(count):
            if i not in indices:
                limits[index].append(self.distribution.set.args[i])
                limits[index] = tuple(limits[index])
                index += 1
        if self.distribution.is_Continuous:
            f = Lambda(sym, integrate(self.distribution(*all_syms), *limits))
        elif self.distribution.is_Discrete:
            f = Lambda(sym, summation(self.distribution(*all_syms), *limits))
        return f.xreplace(replace_dict)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        syms = tuple((self.value[i] for i in range(self.component_count)))
        rvs = rvs or syms
        if not any((i in rvs for i in syms)):
            return expr
        expr = expr * self.pdf
        for rv in rvs:
            if isinstance(rv, Indexed):
                expr = expr.xreplace({rv: Indexed(str(rv.base), rv.args[1])})
            elif isinstance(rv, RandomSymbol):
                expr = expr.xreplace({rv: rv.symbol})
        if self.value in random_symbols(expr):
            raise NotImplementedError(filldedent('\n            Expectations of expression with unindexed joint random symbols\n            cannot be calculated yet.'))
        limits = tuple(((Indexed(str(rv.base), rv.args[1]), self.distribution.set.args[rv.args[1]]) for rv in syms))
        return Integral(expr, *limits)

    def where(self, condition):
        raise NotImplementedError()

    def compute_density(self, expr):
        raise NotImplementedError()

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {RandomSymbol(self.symbol, self): self.distribution.sample(size, library=library, seed=seed)}

    def probability(self, condition):
        raise NotImplementedError()