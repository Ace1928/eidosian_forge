from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _to_ptm(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the PTM representation."""
    if rep == 'PTM':
        return data
    _check_nqubit_dim(input_dim, output_dim)
    if rep == 'Operator':
        return _from_operator('PTM', data, input_dim, output_dim)
    if rep != 'SuperOp':
        data = _to_superop(rep, data, input_dim, output_dim)
    return _superop_to_ptm(data, input_dim)