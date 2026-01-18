from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _chi_to_choi(data, input_dim):
    """Transform Chi representation to a Choi representation."""
    num_qubits = int(np.log2(input_dim))
    return _transform_from_pauli(data, num_qubits)