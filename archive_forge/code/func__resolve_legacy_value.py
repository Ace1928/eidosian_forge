from collections.abc import MutableSequence
import qiskit._accelerate.quantum_circuit
from .exceptions import CircuitError
from .instruction import Instruction
from .operation import Operation
def _resolve_legacy_value(self, operation, qargs, cargs) -> CircuitInstruction:
    """Resolve the old-style 3-tuple into the new :class:`CircuitInstruction` type."""
    if not isinstance(operation, Operation) and hasattr(operation, 'to_instruction'):
        operation = operation.to_instruction()
    if not isinstance(operation, Operation):
        raise CircuitError('object is not an Operation.')
    expanded_qargs = [self._circuit.qbit_argument_conversion(qarg) for qarg in qargs or []]
    expanded_cargs = [self._circuit.cbit_argument_conversion(carg) for carg in cargs or []]
    if isinstance(operation, Instruction):
        broadcast_args = list(operation.broadcast_arguments(expanded_qargs, expanded_cargs))
    else:
        broadcast_args = list(Instruction.broadcast_arguments(operation, expanded_qargs, expanded_cargs))
    if len(broadcast_args) > 1:
        raise CircuitError('QuantumCircuit.data modification does not support argument broadcasting.')
    qargs, cargs = broadcast_args[0]
    self._circuit._check_dups(qargs)
    return CircuitInstruction(operation, tuple(qargs), tuple(cargs))