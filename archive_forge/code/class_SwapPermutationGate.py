import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
@value.value_equality
class SwapPermutationGate(PermutationGate):
    """Generic swap gate."""

    def __init__(self, swap_gate: 'cirq.Gate'=ops.SWAP):
        super().__init__(2, swap_gate)

    def permutation(self) -> Dict[int, int]:
        return {0: 1, 1: 0}

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        yield self.swap_gate(*qubits)

    def __repr__(self) -> str:
        return 'cirq.contrib.acquaintance.SwapPermutationGate(' + ('' if self.swap_gate == ops.SWAP else repr(self.swap_gate)) + ')'

    def _value_equality_values_(self) -> Any:
        return (self.swap_gate,)

    def _commutes_(self, other: Any, *, atol: float=1e-08) -> Union[bool, NotImplementedType]:
        if isinstance(other, ops.Gate) and isinstance(other, ops.InterchangeableQubitsGate) and (protocols.num_qubits(other) == 2):
            return True
        return NotImplemented