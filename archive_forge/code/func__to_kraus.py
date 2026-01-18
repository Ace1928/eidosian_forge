from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _to_kraus(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Kraus representation."""
    if rep == 'Kraus':
        return data
    if rep == 'Stinespring':
        return _stinespring_to_kraus(data, output_dim)
    if rep == 'Operator':
        return _from_operator('Kraus', data, input_dim, output_dim)
    if rep != 'Choi':
        data = _to_choi(rep, data, input_dim, output_dim)
    return _choi_to_kraus(data, input_dim, output_dim)