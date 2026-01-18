from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def _compute_commutator_so3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes the commutator of the SO(3)-matrices ``a`` and ``b``.

    The computation uses the fact that the inverse of an SO(3)-matrix is equal to its transpose.

    Args:
        a: SO(3)-matrix
        b: SO(3)-matrix

    Returns:
        The commutator [a,b] of ``a`` and ``b`` w

    Raises:
        ValueError: if at least one of ``a`` or ``b`` is not an SO(3)-matrix.
    """
    _check_is_so3(a)
    _check_is_so3(b)
    a_dagger = np.conj(a).T
    b_dagger = np.conj(b).T
    return np.dot(np.dot(np.dot(a, b), a_dagger), b_dagger)