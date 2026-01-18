import json
import urllib.parse
from typing import List, cast, Tuple, Any, Iterable
from cirq import ops, circuits, devices, protocols
from cirq.contrib.quirk.linearize_circuit import linearize_circuit_qubits
from cirq.contrib.quirk.quirk_gate import (
def _to_quirk_cols(op: ops.Operation, prefer_unknown_gate_to_failure: bool) -> Iterable[Tuple[List[Any], bool]]:
    gate = _try_convert_to_quirk_gate(op, prefer_unknown_gate_to_failure)
    qubits = cast(Iterable[devices.LineQubit], op.qubits)
    max_index = max((q.x for q in qubits))
    col = [1] * (max_index + 1)
    for i, q in enumerate(qubits):
        col[q.x] = gate.keys[min(i, len(gate.keys) - 1)]
    yield (col, gate.can_merge)