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
@classmethod
def _sample_scipy(cls, dist, size, seed):
    """Sample from SciPy."""
    from scipy import stats as scipy_stats
    import numpy
    scipy_rv_map = {'WishartDistribution': lambda dist, size, rand_state: scipy_stats.wishart.rvs(df=int(dist.n), scale=matrix2numpy(dist.scale_matrix, float), size=size), 'MatrixNormalDistribution': lambda dist, size, rand_state: scipy_stats.matrix_normal.rvs(mean=matrix2numpy(dist.location_matrix, float), rowcov=matrix2numpy(dist.scale_matrix_1, float), colcov=matrix2numpy(dist.scale_matrix_2, float), size=size, random_state=rand_state)}
    sample_shape = {'WishartDistribution': lambda dist: dist.scale_matrix.shape, 'MatrixNormalDistribution': lambda dist: dist.location_matrix.shape}
    dist_list = scipy_rv_map.keys()
    if dist.__class__.__name__ not in dist_list:
        return None
    if seed is None or isinstance(seed, int):
        rand_state = numpy.random.default_rng(seed=seed)
    else:
        rand_state = seed
    samp = scipy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
    return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))