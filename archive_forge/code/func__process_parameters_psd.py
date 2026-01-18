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
def _process_parameters_psd(self, dim, mean, cov):
    if dim is None:
        if mean is None:
            if cov is None:
                dim = 1
            else:
                cov = np.asarray(cov, dtype=float)
                if cov.ndim < 2:
                    dim = 1
                else:
                    dim = cov.shape[0]
        else:
            mean = np.asarray(mean, dtype=float)
            dim = mean.size
    elif not np.isscalar(dim):
        raise ValueError('Dimension of random variable must be a scalar.')
    if mean is None:
        mean = np.zeros(dim)
    mean = np.asarray(mean, dtype=float)
    if cov is None:
        cov = 1.0
    cov = np.asarray(cov, dtype=float)
    if dim == 1:
        mean = mean.reshape(1)
        cov = cov.reshape(1, 1)
    if mean.ndim != 1 or mean.shape[0] != dim:
        raise ValueError("Array 'mean' must be a vector of length %d." % dim)
    if cov.ndim == 0:
        cov = cov * np.eye(dim)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    elif cov.ndim == 2 and cov.shape != (dim, dim):
        rows, cols = cov.shape
        if rows != cols:
            msg = "Array 'cov' must be square if it is two dimensional, but cov.shape = %s." % str(cov.shape)
        else:
            msg = "Dimension mismatch: array 'cov' is of shape %s, but 'mean' is a vector of length %d."
            msg = msg % (str(cov.shape), len(mean))
        raise ValueError(msg)
    elif cov.ndim > 2:
        raise ValueError("Array 'cov' must be at most two-dimensional, but cov.ndim = %d" % cov.ndim)
    return (dim, mean, cov)