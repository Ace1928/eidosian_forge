from math import prod
from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
from sympy.external import import_module
class SampleMatrixPymc:
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
        pymc_rv_map = {'MatrixNormalDistribution': lambda dist: pymc.MatrixNormal('X', mu=matrix2numpy(dist.location_matrix, float), rowcov=matrix2numpy(dist.scale_matrix_1, float), colcov=matrix2numpy(dist.scale_matrix_2, float), shape=dist.location_matrix.shape), 'WishartDistribution': lambda dist: pymc.WishartBartlett('X', nu=int(dist.n), S=matrix2numpy(dist.scale_matrix, float))}
        sample_shape = {'WishartDistribution': lambda dist: dist.scale_matrix.shape, 'MatrixNormalDistribution': lambda dist: dist.location_matrix.shape}
        dist_list = pymc_rv_map.keys()
        if dist.__class__.__name__ not in dist_list:
            return None
        import logging
        logging.getLogger('pymc').setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samps = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)['X']
        return samps.reshape(size + sample_shape[dist.__class__.__name__](dist))