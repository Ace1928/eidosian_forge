from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _to_operator(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Operator representation."""
    if rep == 'Operator':
        return data
    if rep == 'Stinespring':
        return _stinespring_to_operator(data, output_dim)
    if rep != 'Kraus':
        data = _to_kraus(rep, data, input_dim, output_dim)
    return _kraus_to_operator(data)