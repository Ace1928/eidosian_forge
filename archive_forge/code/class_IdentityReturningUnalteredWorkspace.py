import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
class IdentityReturningUnalteredWorkspace:

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        return args.available_buffer

    def _unitary_(self):
        return np.eye(2)

    def _num_qubits_(self):
        return 1