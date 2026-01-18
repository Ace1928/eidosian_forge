from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def _decompose_abc(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Decomposes 2x2 unitary matrix.

    Returns 2x2 special unitary matrices A, B, C and phase delta, such that:
    * ABC = I.
    * AXBXC * exp(1j*delta) = matrix.

    See [1], chapter 4.
    """
    assert matrix.shape == (2, 2)
    delta = np.angle(np.linalg.det(matrix)) * 0.5
    alpha = np.angle(matrix[0, 0]) + np.angle(matrix[0, 1]) - 2 * delta
    beta = np.angle(matrix[0, 0]) - np.angle(matrix[0, 1])
    m00_abs = np.abs(matrix[0, 0])
    if np.abs(m00_abs - 1.0) < 1e-09:
        m00_abs = 1
    theta = 2 * np.arccos(m00_abs)
    a = unitary(ops.rz(-alpha)) @ unitary(ops.ry(-theta / 2))
    b = unitary(ops.ry(theta / 2)) @ unitary(ops.rz((alpha + beta) / 2))
    c = unitary(ops.rz((alpha - beta) / 2))
    x = unitary(ops.X)
    assert np.allclose(a @ b @ c, np.eye(2), atol=0.01)
    assert np.allclose(a @ x @ b @ x @ c * np.exp(1j * delta), matrix, atol=0.01)
    return (a, b, c, delta)