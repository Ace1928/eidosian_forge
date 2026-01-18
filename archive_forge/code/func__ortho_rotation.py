import warnings
from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import fast_logdet, randomized_svd, squared_norm
from ..utils.validation import check_is_fitted
def _ortho_rotation(components, method='varimax', tol=1e-06, max_iter=100):
    """Return rotated components."""
    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0
    for _ in range(max_iter):
        comp_rot = np.dot(components, rotation_matrix)
        if method == 'varimax':
            tmp = comp_rot * np.transpose((comp_rot ** 2).sum(axis=0) / nrow)
        elif method == 'quartimax':
            tmp = 0
        u, s, v = np.linalg.svd(np.dot(components.T, comp_rot ** 3 - tmp))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new
    return np.dot(components, rotation_matrix).T