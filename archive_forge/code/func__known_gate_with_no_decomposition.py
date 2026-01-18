from typing import Any
import numpy as np
from cirq import devices, protocols, ops, circuits
from cirq.testing import lin_alg_utils
def _known_gate_with_no_decomposition(val: Any):
    """Checks whether `val` is a known gate with no default decomposition to default gateset."""
    if isinstance(val, ops.MatrixGate):
        return protocols.qid_shape(val) not in [(2,), (2,) * 2, (2,) * 3]
    if isinstance(val, ops.BaseDensePauliString) and (not protocols.has_unitary(val)):
        return True
    if isinstance(val, ops.ControlledGate):
        if protocols.is_parameterized(val):
            return True
        if isinstance(val.sub_gate, ops.MatrixGate) and protocols.num_qubits(val.sub_gate) > 1:
            return True
        if val.control_qid_shape != (2,) * val.num_controls():
            return True
        if isinstance(val.control_values, ops.SumOfProducts):
            return True
        return _known_gate_with_no_decomposition(val.sub_gate)
    return False