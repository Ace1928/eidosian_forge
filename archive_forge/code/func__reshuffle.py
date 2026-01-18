from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _reshuffle(mat, shape):
    """Reshuffle the indices of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)), (shape[3] * shape[1], shape[0] * shape[2]))