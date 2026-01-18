from typing import Any, TypeVar, Optional
import numpy as np
from typing_extensions import Protocol
from cirq import qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
def _strat_has_unitary_from_has_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _has_unitary_ method."""
    if hasattr(val, '_has_unitary_'):
        result = val._has_unitary_()
        if result is NotImplemented:
            return None
        return result
    return None