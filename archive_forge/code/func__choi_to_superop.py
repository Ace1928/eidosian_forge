from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _choi_to_superop(data, input_dim, output_dim):
    """Transform Choi to SuperOp representation."""
    shape = (input_dim, output_dim, input_dim, output_dim)
    return _reshuffle(data, shape)