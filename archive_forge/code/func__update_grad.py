import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
def _update_grad(self):
    if not self.g_updated:
        self._update_grad_impl()
        self.g_updated = True