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
def _dirichlet_multinomial_check_parameters(alpha, n, x=None):
    alpha = np.asarray(alpha)
    n = np.asarray(n)
    if x is not None:
        try:
            x, alpha = np.broadcast_arrays(x, alpha)
        except ValueError as e:
            msg = '`x` and `alpha` must be broadcastable.'
            raise ValueError(msg) from e
        x_int = np.floor(x)
        if np.any(x < 0) or np.any(x != x_int):
            raise ValueError('`x` must contain only non-negative integers.')
        x = x_int
    if np.any(alpha <= 0):
        raise ValueError('`alpha` must contain only positive values.')
    n_int = np.floor(n)
    if np.any(n <= 0) or np.any(n != n_int):
        raise ValueError('`n` must be a positive integer.')
    n = n_int
    sum_alpha = np.sum(alpha, axis=-1)
    sum_alpha, n = np.broadcast_arrays(sum_alpha, n)
    return (alpha, sum_alpha, n) if x is None else (alpha, sum_alpha, n, x)