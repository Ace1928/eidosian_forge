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
def _log_norm_factor(self, dim, kappa):
    halfdim = 0.5 * dim
    return 0.5 * (dim - 2) * np.log(kappa) - halfdim * _LOG_2PI - np.log(ive(halfdim - 1, kappa)) - kappa