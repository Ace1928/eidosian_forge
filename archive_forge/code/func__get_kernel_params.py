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
def _get_kernel_params(self):
    params = self.kernel_params
    if params is None:
        params = {}
    if not callable(self.kernel) and self.kernel != 'precomputed':
        for param in KERNEL_PARAMS[self.kernel]:
            if getattr(self, param) is not None:
                params[param] = getattr(self, param)
    elif self.gamma is not None or self.coef0 is not None or self.degree is not None:
        raise ValueError("Don't pass gamma, coef0 or degree to Nystroem if using a callable or precomputed kernel")
    return params