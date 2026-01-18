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
def _rvs_2d(self, mu, kappa, size, random_state):
    """
        In 2D, the von Mises-Fisher distribution reduces to the
        von Mises distribution which can be efficiently sampled by numpy.
        This method is much faster than the general rejection
        sampling based algorithm.

        """
    mean_angle = np.arctan2(mu[1], mu[0])
    angle_samples = random_state.vonmises(mean_angle, kappa, size=size)
    samples = np.stack([np.cos(angle_samples), np.sin(angle_samples)], axis=-1)
    return samples