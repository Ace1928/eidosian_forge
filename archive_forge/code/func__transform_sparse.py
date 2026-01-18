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
def _transform_sparse(X, sample_steps, sample_interval):
    indices = X.indices.copy()
    indptr = X.indptr.copy()
    data_step = np.sqrt(X.data * sample_interval)
    X_step = sp.csr_matrix((data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False)
    X_new = [X_step]
    log_step_nz = sample_interval * np.log(X.data)
    step_nz = 2 * X.data * sample_interval
    for j in range(1, sample_steps):
        factor_nz = np.sqrt(step_nz / np.cosh(np.pi * j * sample_interval))
        data_step = factor_nz * np.cos(j * log_step_nz)
        X_step = sp.csr_matrix((data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False)
        X_new.append(X_step)
        data_step = factor_nz * np.sin(j * log_step_nz)
        X_step = sp.csr_matrix((data_step, indices, indptr), shape=X.shape, dtype=X.dtype, copy=False)
        X_new.append(X_step)
    return sp.hstack(X_new)