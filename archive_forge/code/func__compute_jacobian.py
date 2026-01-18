import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def _compute_jacobian(self, J_eq, J_ineq, s):
    if self.n_ineq == 0:
        return J_eq
    elif sps.issparse(J_eq) or sps.issparse(J_ineq):
        J_eq = sps.csr_matrix(J_eq)
        J_ineq = sps.csr_matrix(J_ineq)
        return self._assemble_sparse_jacobian(J_eq, J_ineq, s)
    else:
        S = np.diag(s)
        zeros = np.zeros((self.n_eq, self.n_ineq))
        if sps.issparse(J_ineq):
            J_ineq = J_ineq.toarray()
        if sps.issparse(J_eq):
            J_eq = J_eq.toarray()
        return np.block([[J_eq, zeros], [J_ineq, S]])