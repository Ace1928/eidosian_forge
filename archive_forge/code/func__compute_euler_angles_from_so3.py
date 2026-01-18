from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
def _compute_euler_angles_from_so3(matrix: np.ndarray) -> tuple[float, float, float]:
    """Computes the Euler angles from the SO(3)-matrix u.

    Uses the algorithm from Gregory Slabaugh,
    see `here <https://www.gregslabaugh.net/publications/euler.pdf>`_.

    Args:
        matrix: The SO(3)-matrix for which the Euler angles need to be computed.

    Returns:
        Tuple (phi, theta, psi), where phi is rotation about z-axis, theta rotation about y-axis
        and psi rotation about x-axis.
    """
    matrix = np.round(matrix, decimals=10)
    if matrix[2][0] != 1 and matrix[2][1] != -1:
        theta = -math.asin(matrix[2][0])
        psi = math.atan2(matrix[2][1] / math.cos(theta), matrix[2][2] / math.cos(theta))
        phi = math.atan2(matrix[1][0] / math.cos(theta), matrix[0][0] / math.cos(theta))
        return (phi, theta, psi)
    else:
        phi = 0
        if matrix[2][0] == 1:
            theta = math.pi / 2
            psi = phi + math.atan2(matrix[0][1], matrix[0][2])
        else:
            theta = -math.pi / 2
            psi = -phi + math.atan2(-matrix[0][1], -matrix[0][2])
        return (phi, theta, psi)