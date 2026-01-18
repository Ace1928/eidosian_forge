from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
def _check_is_so3(matrix: np.ndarray) -> None:
    """Check whether ``matrix`` is SO(3), otherwise raise an error."""
    if matrix.shape != (3, 3):
        raise ValueError(f'Matrix must have shape (3, 3) but has {matrix.shape}.')
    if abs(np.linalg.det(matrix) - 1) > 0.0001:
        raise ValueError(f'Determinant of matrix must be 1, but is {np.linalg.det(matrix)}.')