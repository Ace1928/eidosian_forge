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
def _dirichlet_check_input(alpha, x):
    x = np.asarray(x)
    if x.shape[0] + 1 != alpha.shape[0] and x.shape[0] != alpha.shape[0]:
        raise ValueError(f"Vector 'x' must have either the same number of entries as, or one entry fewer than, parameter vector 'a', but alpha.shape = {alpha.shape} and x.shape = {x.shape}.")
    if x.shape[0] != alpha.shape[0]:
        xk = np.array([1 - np.sum(x, 0)])
        if xk.ndim == 1:
            x = np.append(x, xk)
        elif xk.ndim == 2:
            x = np.vstack((x, xk))
        else:
            raise ValueError('The input must be one dimensional or a two dimensional matrix containing the entries.')
    if np.min(x) < 0:
        raise ValueError("Each entry in 'x' must be greater than or equal to zero.")
    if np.max(x) > 1:
        raise ValueError("Each entry in 'x' must be smaller or equal one.")
    xeq0 = x == 0
    alphalt1 = alpha < 1
    if x.shape != alpha.shape:
        alphalt1 = np.repeat(alphalt1, x.shape[-1], axis=-1).reshape(x.shape)
    chk = np.logical_and(xeq0, alphalt1)
    if np.sum(chk):
        raise ValueError("Each entry in 'x' must be greater than zero if its alpha is less than one.")
    if (np.abs(np.sum(x, 0) - 1.0) > 1e-09).any():
        raise ValueError("The input vector 'x' must lie within the normal simplex. but np.sum(x, 0) = %s." % np.sum(x, 0))
    return x