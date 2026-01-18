from __future__ import annotations
from collections.abc import Sequence
import math
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
def _compute_su2_from_euler_angles(angles: tuple[float, float, float]) -> np.ndarray:
    """Computes SU(2)-matrix from Euler angles.

    Args:
        angles: The tuple containing the Euler angles for which the corresponding SU(2)-matrix
            needs to be computed.

    Returns:
        The SU(2)-matrix corresponding to the Euler angles in angles.
    """
    phi, theta, psi = angles
    uz_phi = np.array([[np.exp(-0.5j * phi), 0], [0, np.exp(0.5j * phi)]], dtype=complex)
    uy_theta = np.array([[math.cos(theta / 2), math.sin(theta / 2)], [-math.sin(theta / 2), math.cos(theta / 2)]], dtype=complex)
    ux_psi = np.array([[math.cos(psi / 2), math.sin(psi / 2) * 1j], [math.sin(psi / 2) * 1j, math.cos(psi / 2)]], dtype=complex)
    return np.dot(uz_phi, np.dot(uy_theta, ux_psi))