import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
class IdentityVectorFunction(LinearVectorFunction):
    """Identity vector function and its derivatives.

    The Jacobian is the identity matrix, returned as a dense array when
    `sparse_jacobian=False` and as a csr matrix otherwise. The Hessian is
    identically zero and it is returned as a csr matrix.
    """

    def __init__(self, x0, sparse_jacobian):
        n = len(x0)
        if sparse_jacobian or sparse_jacobian is None:
            A = sps.eye(n, format='csr')
            sparse_jacobian = True
        else:
            A = np.eye(n)
            sparse_jacobian = False
        super().__init__(A, x0, sparse_jacobian)