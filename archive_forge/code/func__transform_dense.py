import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from .base import (
from .metrics.pairwise import KERNEL_PARAMS, PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from .utils import check_random_state, deprecated
from .utils._param_validation import Interval, StrOptions
from .utils.extmath import safe_sparse_dot
from .utils.validation import (
@staticmethod
def _transform_dense(X, sample_steps, sample_interval):
    non_zero = X != 0.0
    X_nz = X[non_zero]
    X_step = np.zeros_like(X)
    X_step[non_zero] = np.sqrt(X_nz * sample_interval)
    X_new = [X_step]
    log_step_nz = sample_interval * np.log(X_nz)
    step_nz = 2 * X_nz * sample_interval
    for j in range(1, sample_steps):
        factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))
        X_step = np.zeros_like(X)
        X_step[non_zero] = factor_nz * np.cos(j * log_step_nz)
        X_new.append(X_step)
        X_step = np.zeros_like(X)
        X_step[non_zero] = factor_nz * np.sin(j * log_step_nz)
        X_new.append(X_step)
    return np.hstack(X_new)