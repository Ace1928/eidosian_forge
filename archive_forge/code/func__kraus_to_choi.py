from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _kraus_to_choi(data):
    """Transform Kraus representation to Choi representation."""
    choi = 0
    kraus_l, kraus_r = data
    if kraus_r is None:
        for i in kraus_l:
            vec = i.ravel(order='F')
            choi += np.outer(vec, vec.conj())
    else:
        for i, j in zip(kraus_l, kraus_r):
            choi += np.outer(i.ravel(order='F'), j.ravel(order='F').conj())
    return choi