import pytest
import cirq
class NotImplementedOperation(cirq.Operation):

    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        return cirq.LineQubit.range(2)