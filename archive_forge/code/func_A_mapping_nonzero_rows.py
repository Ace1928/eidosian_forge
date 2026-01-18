from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def A_mapping_nonzero_rows(problem_data_tensor, var_length):
    problem_data_tensor_csc = problem_data_tensor.tocsc()
    A_nrows = problem_data_tensor.shape[0] // (var_length + 1)
    A_ncols = var_length
    A_mapping = problem_data_tensor_csc[:A_nrows * A_ncols, :-1]
    A_mapping_nonzero_rows, _ = A_mapping.nonzero()
    return np.unique(A_mapping_nonzero_rows)