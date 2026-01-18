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
class SampleJointScipy:
    """Returns the sample from scipy of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""
        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        from scipy import stats as scipy_stats
        scipy_rv_map = {'MultivariateNormalDistribution': lambda dist, size: scipy_stats.multivariate_normal.rvs(mean=matrix2numpy(dist.mu).flatten(), cov=matrix2numpy(dist.sigma), size=size, random_state=rand_state), 'MultivariateBetaDistribution': lambda dist, size: scipy_stats.dirichlet.rvs(alpha=list2numpy(dist.alpha, float).flatten(), size=size, random_state=rand_state), 'MultinomialDistribution': lambda dist, size: scipy_stats.multinomial.rvs(n=int(dist.n), p=list2numpy(dist.p, float).flatten(), size=size, random_state=rand_state)}
        sample_shape = {'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape, 'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape, 'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape}
        dist_list = scipy_rv_map.keys()
        if dist.__class__.__name__ not in dist_list:
            return None
        samples = scipy_rv_map[dist.__class__.__name__](dist, size)
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))