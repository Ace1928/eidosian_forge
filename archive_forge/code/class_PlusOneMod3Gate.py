import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class PlusOneMod3Gate(cirq.testing.SingleQubitGate):

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])