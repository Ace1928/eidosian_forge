import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
def _pass_operation_over(pauli_map: Dict[TKey, pauli_gates.Pauli], op: 'cirq.Operation', after_to_before: bool=False) -> bool:
    if isinstance(op, gate_operation.GateOperation):
        gate = op.gate
        if isinstance(gate, clifford_gate.SingleQubitCliffordGate):
            return _pass_single_clifford_gate_over(pauli_map, gate, cast(TKey, op.qubits[0]), after_to_before=after_to_before)
        if isinstance(gate, pauli_interaction_gate.PauliInteractionGate):
            return _pass_pauli_interaction_gate_over(pauli_map, gate, cast(TKey, op.qubits[0]), cast(TKey, op.qubits[1]), after_to_before=after_to_before)
    raise NotImplementedError(f'Unsupported operation: {op!r}')