import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
class NoEffect:

    def _num_qubits_(self):
        return 1

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        return NotImplemented