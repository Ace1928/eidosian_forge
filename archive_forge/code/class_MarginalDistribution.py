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
class MarginalDistribution(Distribution):
    """
    Represents the marginal distribution of a joint probability space.

    Initialised using a probability distribution and random variables(or
    their indexed components) which should be a part of the resultant
    distribution.
    """

    def __new__(cls, dist, *rvs):
        if len(rvs) == 1 and iterable(rvs[0]):
            rvs = tuple(rvs[0])
        if not all((isinstance(rv, (Indexed, RandomSymbol)) for rv in rvs)):
            raise ValueError(filldedent('Marginal distribution can be\n             intitialised only in terms of random variables or indexed random\n             variables'))
        rvs = Tuple.fromiter((rv for rv in rvs))
        if not isinstance(dist, JointDistribution) and len(random_symbols(dist)) == 0:
            return dist
        return Basic.__new__(cls, dist, rvs)

    def check(self):
        pass

    @property
    def set(self):
        rvs = [i for i in self.args[1] if isinstance(i, RandomSymbol)]
        return ProductSet(*[rv.pspace.set for rv in rvs])

    @property
    def symbols(self):
        rvs = self.args[1]
        return {rv.pspace.symbol for rv in rvs}

    def pdf(self, *x):
        expr, rvs = (self.args[0], self.args[1])
        marginalise_out = [i for i in random_symbols(expr) if i not in rvs]
        if isinstance(expr, JointDistribution):
            count = len(expr.domain.args)
            x = Dummy('x', real=True)
            syms = tuple((Indexed(x, i) for i in count))
            expr = expr.pdf(syms)
        else:
            syms = tuple((rv.pspace.symbol if isinstance(rv, RandomSymbol) else rv.args[0] for rv in rvs))
        return Lambda(syms, self.compute_pdf(expr, marginalise_out))(*x)

    def compute_pdf(self, expr, rvs):
        for rv in rvs:
            lpdf = 1
            if isinstance(rv, RandomSymbol):
                lpdf = rv.pspace.pdf
            expr = self.marginalise_out(expr * lpdf, rv)
        return expr

    def marginalise_out(self, expr, rv):
        from sympy.concrete.summations import Sum
        if isinstance(rv, RandomSymbol):
            dom = rv.pspace.set
        elif isinstance(rv, Indexed):
            dom = rv.base.component_domain(rv.pspace.component_domain(rv.args[1]))
        expr = expr.xreplace({rv: rv.pspace.symbol})
        if rv.pspace.is_Continuous:
            expr = Integral(expr, (rv.pspace.symbol, dom))
        elif rv.pspace.is_Discrete:
            if dom in (S.Integers, S.Naturals, S.Naturals0):
                dom = (dom.inf, dom.sup)
            expr = Sum(expr, (rv.pspace.symbol, dom))
        return expr

    def __call__(self, *args):
        return self.pdf(*args)