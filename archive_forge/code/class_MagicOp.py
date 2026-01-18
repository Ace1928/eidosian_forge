import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
class MagicOp(cirq.Operation):

    def __init__(self, *qubits):
        self._qubits = qubits

    def with_qubits(self, *new_qubits):
        return MagicOp(*new_qubits)

    @property
    def qubits(self):
        return self._qubits

    def __str__(self):
        return 'MagicOperate'