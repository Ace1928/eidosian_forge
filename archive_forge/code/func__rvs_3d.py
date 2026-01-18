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
def _rvs_3d(self, kappa, size, random_state):
    """
        Generate samples from a von Mises-Fisher distribution
        with mu = [1, 0, 0] and kappa. Samples then have to be
        rotated towards the desired mean direction mu.
        This method is much faster than the general rejection
        sampling based algorithm.
        Reference: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf

        """
    if size is None:
        sample_size = 1
    else:
        sample_size = size
    x = random_state.random(sample_size)
    x = 1.0 + np.log(x + (1.0 - x) * np.exp(-2 * kappa)) / kappa
    temp = np.sqrt(1.0 - np.square(x))
    uniformcircle = _sample_uniform_direction(2, sample_size, random_state)
    samples = np.stack([x, temp * uniformcircle[..., 0], temp * uniformcircle[..., 1]], axis=-1)
    if size is None:
        samples = np.squeeze(samples)
    return samples