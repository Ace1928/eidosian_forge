import math
import warnings
from functools import lru_cache
from scipy.spatial import KDTree
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript
def _group_commutator_decompose(matrix, tol=1e-05):
    """Performs a group commutator decomposition :math:`U = V' \\times W' \\times V'^{\\dagger} \\times W'^{\\dagger}`
    as given in the Section 4.1 of `arXiv:0505030 <https://arxiv.org/abs/quant-ph/0505030>`_."""
    quaternion = _quaternion_transform(matrix)
    theta, axis = (2 * qml.math.arccos(qml.math.clip(quaternion[0], -1.0, 1.0)), quaternion[1:])
    if qml.math.allclose(axis, 0.0, atol=tol) and qml.math.isclose(theta % math.pi, 0.0, atol=tol):
        return (qml.math.eye(2, dtype=complex), qml.math.eye(2, dtype=complex))
    phi = 2.0 * qml.math.arcsin(qml.math.sqrt(qml.math.sqrt(0.5 - 0.5 * qml.math.cos(theta / 2))))
    v = qml.RX(phi, [0])
    w = qml.RY(2 * math.pi - phi, [0]) if axis[2] > 0 else qml.RY(phi, [0])
    ud = qml.math.linalg.eig(matrix)[1]
    vwd = qml.math.linalg.eig(qml.matrix(v @ w @ v.adjoint() @ w.adjoint()))[1]
    s = ud @ qml.math.conj(qml.math.transpose(vwd))
    sdg = vwd @ qml.math.conj(qml.math.transpose(ud))
    v_hat = s @ v.matrix() @ sdg
    w_hat = s @ w.matrix() @ sdg
    return (w_hat, v_hat)