from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _to_superop(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the SuperOp representation."""
    if rep == 'SuperOp':
        return data
    if rep == 'Operator':
        return _from_operator('SuperOp', data, input_dim, output_dim)
    if rep == 'Choi':
        return _choi_to_superop(data, input_dim, output_dim)
    if rep == 'Kraus':
        return _kraus_to_superop(data)
    if rep == 'Chi':
        data = _chi_to_choi(data, input_dim)
        return _choi_to_superop(data, input_dim, output_dim)
    if rep == 'PTM':
        return _ptm_to_superop(data, input_dim)
    if rep == 'Stinespring':
        return _stinespring_to_superop(data, input_dim, output_dim)
    raise QiskitError(f'Invalid QuantumChannel {rep}')