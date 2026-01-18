from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _kraus_to_superop(data):
    """Transform Kraus representation to SuperOp representation."""
    kraus_l, kraus_r = data
    superop = 0
    if kraus_r is None:
        for i in kraus_l:
            superop += np.kron(np.conj(i), i)
    else:
        for i, j in zip(kraus_l, kraus_r):
            superop += np.kron(np.conj(j), i)
    return superop