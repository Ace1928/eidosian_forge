import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def _sample_uniform_direction(dim, size, random_state):
    """
    Private method to generate uniform directions
    Reference: Marsaglia, G. (1972). "Choosing a Point from the Surface of a
               Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.
    """
    samples_shape = np.append(size, dim)
    samples = random_state.standard_normal(samples_shape)
    samples /= np.linalg.norm(samples, axis=-1, keepdims=True)
    return samples