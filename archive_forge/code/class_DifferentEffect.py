import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
class DifferentEffect:

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
        o = args.subspace_index(0)
        i = args.subspace_index(1)
        args.available_buffer[o] = args.target_tensor[i]
        args.available_buffer[i] = args.target_tensor[o]
        return args.available_buffer

    def _unitary_(self):
        return np.eye(2, dtype=np.complex128)

    def _num_qubits_(self):
        return 1