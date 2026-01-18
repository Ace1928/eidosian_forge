from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def _compute_rotation_between(from_vector: np.ndarray, to_vector: np.ndarray) -> np.ndarray:
    """Computes the SO(3)-matrix for rotating ``from_vector`` to ``to_vector``.

    Args:
        from_vector: unit vector of size 3
        to_vector: unit vector of size 3

    Returns:
        SO(3)-matrix that brings ``from_vector`` to ``to_vector``.

    Raises:
        ValueError: if at least one of ``from_vector`` of ``to_vector`` is not a 3-dim unit vector.
    """
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)
    dot = np.dot(from_vector, to_vector)
    cross = _cross_product_matrix(np.cross(from_vector, to_vector))
    rotation_matrix = np.identity(3) + cross + np.dot(cross, cross) / (1 + dot)
    return rotation_matrix