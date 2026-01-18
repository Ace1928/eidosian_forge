import abc
from typing import (
from cirq import ops, value, devices
@classmethod
def _replace_qubits(cls, old_qubits: Iterable['cirq.Qid'], qubits: List['cirq.Qid']) -> Tuple['cirq.Qid', ...]:
    return tuple((Cell._replace_qubit(e, qubits) for e in old_qubits))