import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
def _commutes_(self, other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType]:
    if isinstance(other, ops.Gate) and isinstance(other, ops.InterchangeableQubitsGate) and (protocols.num_qubits(other) == 2):
        return True
    return NotImplemented