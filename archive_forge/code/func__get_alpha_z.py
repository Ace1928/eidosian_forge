import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def _get_alpha_z(omega, n, k):
    """Computes the rotation angles required to implement the uniformly-controlled Z rotation
    applied to the :math:`k`th qubit.

    The :math:`j`th angle is related to the phases omega of the desired amplitudes via:

    .. math:: \\alpha^{z,k}_j = \\sum_{l=1}^{2^{k-1}} \\frac{\\omega_{(2j-1) 2^{k-1}+l} - \\omega_{(2j-2) 2^{k-1}+l}}{2^{k-1}}

    Args:
        omega (tensor_like): phases of the state to prepare
        n (int): total number of qubits for the uniformly-controlled rotation
        k (int): index of current qubit

    Returns:
        array representing :math:`\\alpha^{z,k}`
    """
    indices1 = [[(2 * j - 1) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)] for j in range(1, 2 ** (n - k) + 1)]
    indices2 = [[(2 * j - 2) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)] for j in range(1, 2 ** (n - k) + 1)]
    term1 = qml.math.take(omega, indices=indices1, axis=-1)
    term2 = qml.math.take(omega, indices=indices2, axis=-1)
    diff = (term1 - term2) / 2 ** (k - 1)
    return qml.math.sum(diff, axis=-1)