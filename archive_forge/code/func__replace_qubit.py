import abc
from typing import (
from cirq import ops, value, devices
@classmethod
def _replace_qubit(cls, old_qubit: 'cirq.Qid', qubits: List['cirq.Qid']) -> 'cirq.Qid':
    if not isinstance(old_qubit, devices.LineQubit):
        raise ValueError(f'Can only map from line qubits, but got {old_qubit!r}.')
    if not 0 <= old_qubit.x < len(qubits):
        raise ValueError(f'Line qubit index ({old_qubit.x}) not in range({len(qubits)})')
    return qubits[old_qubit.x]