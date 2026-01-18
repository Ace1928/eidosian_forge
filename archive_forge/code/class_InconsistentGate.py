import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
class InconsistentGate(cirq.Gate):

    def _num_qubits_(self):
        return 2

    def _qid_shape_(self):
        return (1, 2, 3, 4)