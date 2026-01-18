import json
import urllib.parse
from typing import List, cast, Tuple, Any, Iterable
from cirq import ops, circuits, devices, protocols
from cirq.contrib.quirk.linearize_circuit import linearize_circuit_qubits
from cirq.contrib.quirk.quirk_gate import (
def _try_convert_to_quirk_gate(op: ops.Operation, prefer_unknown_gate_to_failure: bool) -> QuirkOp:
    quirk_gate = known_quirk_op_for_operation(op)
    if quirk_gate is not None:
        return quirk_gate
    matrix_op = single_qubit_matrix_gate(protocols.unitary(op, None))
    if matrix_op is not None:
        return matrix_op
    if prefer_unknown_gate_to_failure:
        return UNKNOWN_GATE
    raise TypeError(f'Unrecognized operation: {op!r}')