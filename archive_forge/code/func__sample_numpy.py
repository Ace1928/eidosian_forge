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
def _sample_numpy(cls, dist, size, seed):
    """Sample from NumPy."""
    numpy_rv_map = {}
    sample_shape = {}
    dist_list = numpy_rv_map.keys()
    if dist.__class__.__name__ not in dist_list:
        return None
    import numpy
    if seed is None or isinstance(seed, int):
        rand_state = numpy.random.default_rng(seed=seed)
    else:
        rand_state = seed
    samp = numpy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
    return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))