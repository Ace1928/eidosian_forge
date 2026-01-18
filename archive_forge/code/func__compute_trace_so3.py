from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def _compute_trace_so3(matrix: np.ndarray) -> float:
    """Computes trace of an SO(3)-matrix.

    Args:
        matrix: an SO(3)-matrix

    Returns:
        Trace of ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    _check_is_so3(matrix)
    trace = np.matrix.trace(matrix)
    trace_rounded = min(trace, 3)
    return trace_rounded