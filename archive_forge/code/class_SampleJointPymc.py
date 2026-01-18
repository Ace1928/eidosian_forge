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
class SampleJointPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""
        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        pymc_rv_map = {'MultivariateNormalDistribution': lambda dist: pymc.MvNormal('X', mu=matrix2numpy(dist.mu, float).flatten(), cov=matrix2numpy(dist.sigma, float), shape=(1, dist.mu.shape[0])), 'MultivariateBetaDistribution': lambda dist: pymc.Dirichlet('X', a=list2numpy(dist.alpha, float).flatten()), 'MultinomialDistribution': lambda dist: pymc.Multinomial('X', n=int(dist.n), p=list2numpy(dist.p, float).flatten(), shape=(1, len(dist.p)))}
        sample_shape = {'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape, 'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape, 'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape}
        dist_list = pymc_rv_map.keys()
        if dist.__class__.__name__ not in dist_list:
            return None
        import logging
        logging.getLogger('pymc3').setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samples = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)[:]['X']
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))