from typing import Any, Optional
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.ops.dense_pauli_string import DensePauliString
from cirq._import import LazyLoader
import cirq.protocols.unitary_protocol as unitary_protocol
import cirq.protocols.has_unitary_protocol as has_unitary_protocol
import cirq.protocols.qid_shape_protocol as qid_shape_protocol
import cirq.protocols.decompose_protocol as decompose_protocol
def _strat_has_stabilizer_effect_from_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer whether val has stabilizer effect from its unitary.

    Returns whether unitary of `val` normalizes the Pauli group. Works only for
    2x2 unitaries.
    """
    qid_shape = qid_shape_protocol.qid_shape(val, default=None)
    if qid_shape is None or len(qid_shape) > 3 or qid_shape != (2,) * len(qid_shape) or (not has_unitary_protocol.has_unitary(val)):
        return None
    unitary = unitary_protocol.unitary(val)
    if len(qid_shape) == 1:
        return SingleQubitCliffordGate.from_unitary(unitary) is not None
    for q_idx in range(len(qid_shape)):
        for g in 'XZ':
            pauli_string = ['I'] * len(qid_shape)
            pauli_string[q_idx] = g
            ps = DensePauliString(pauli_string)
            p = ps._unitary_()
            if not pauli_string_decomposition.unitary_to_pauli_string(unitary @ p @ unitary.T.conj()):
                return False
    return True