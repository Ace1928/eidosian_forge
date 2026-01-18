from typing import Any, Optional
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.ops.dense_pauli_string import DensePauliString
from cirq._import import LazyLoader
import cirq.protocols.unitary_protocol as unitary_protocol
import cirq.protocols.has_unitary_protocol as has_unitary_protocol
import cirq.protocols.qid_shape_protocol as qid_shape_protocol
import cirq.protocols.decompose_protocol as decompose_protocol
def _strat_has_stabilizer_effect_from_has_stabilizer_effect(val: Any) -> Optional[bool]:
    """Infer whether val has stabilizer effect via its `_has_stabilizer_effect_` method."""
    if hasattr(val, '_has_stabilizer_effect_'):
        result = val._has_stabilizer_effect_()
        if result is not NotImplemented and result is not None:
            return result
    return None