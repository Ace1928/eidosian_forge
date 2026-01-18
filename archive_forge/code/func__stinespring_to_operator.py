from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def _stinespring_to_operator(data, output_dim):
    """Transform Stinespring representation to Operator representation."""
    trace_dim = data[0].shape[0] // output_dim
    if data[1] is not None or trace_dim != 1:
        raise QiskitError('Channel cannot be converted to Operator representation')
    return data[0]