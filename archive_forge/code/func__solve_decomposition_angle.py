from __future__ import annotations
import math
import numpy as np
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from .gate_sequence import _check_is_so3, GateSequence
def _solve_decomposition_angle(matrix: np.ndarray) -> float:
    """Computes angle for balanced commutator of SO(3)-matrix ``matrix``.

    Computes angle a so that the SO(3)-matrix ``matrix`` can be decomposed
    as commutator [v,w] where v and w are both rotations of a about some axis.
    The computation is done by solving a trigonometric equation using scipy.optimize.fsolve.

    Args:
        matrix: The SO(3)-matrix for which the decomposition angle needs to be computed.

    Returns:
        Angle a so that matrix = [v,w] with v and w rotations of a about some axis.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    from scipy.optimize import fsolve
    _check_is_so3(matrix)
    trace = _compute_trace_so3(matrix)
    angle = math.acos(1 / 2 * (trace - 1))
    lhs = math.sin(angle / 2)

    def objective(phi):
        sin_sq = np.sin(phi / 2) ** 2
        return 2 * sin_sq * np.sqrt(1 - sin_sq ** 2) - lhs
    decomposition_angle = fsolve(objective, angle)[0]
    return decomposition_angle