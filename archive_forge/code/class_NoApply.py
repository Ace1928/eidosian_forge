import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
class NoApply:

    def _apply_channel_(self, args: cirq.ApplyChannelArgs):
        return NotImplemented

    def _kraus_(self):
        return [np.array([[0, 0], [0, 0]])]

    def _num_qubits_(self):
        return 1