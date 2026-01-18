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
def _check_data_vs_dist(self, x, dim):
    if x.shape[-1] != dim:
        raise ValueError("The dimensionality of the last axis of 'x' must match the dimensionality of the von Mises Fisher distribution.")
    if not np.allclose(np.linalg.norm(x, axis=-1), 1.0):
        msg = "'x' must be unit vectors of norm 1 along last dimension."
        raise ValueError(msg)