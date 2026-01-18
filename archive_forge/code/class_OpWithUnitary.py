import numpy as np
import cirq
class OpWithUnitary(EmptyOp):

    def __init__(self, unitary):
        self.unitary = unitary

    def _unitary_(self):
        return self.unitary

    @property
    def qubits(self):
        return cirq.LineQubit.range(self.unitary.shape[0].bit_length() - 1)