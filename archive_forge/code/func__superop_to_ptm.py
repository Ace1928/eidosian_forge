from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _superop_to_ptm(data, input_dim):
    """Transform SuperOp representation to PTM representation."""
    num_qubits = int(np.log2(input_dim))
    return _transform_to_pauli(data, num_qubits)