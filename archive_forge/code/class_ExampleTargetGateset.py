from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
class ExampleTargetGateset(cirq.TwoQubitCompilationTargetGateset):

    def __init__(self):
        super().__init__(cirq.X, cirq.CNOT)

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        q0, q1 = op.qubits
        return [cirq.X.on_each(q0, q1), cirq.CNOT(q0, q1)] * 10

    def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        return cirq.X(*op.qubits) if op.gate == cirq.Y else NotImplemented