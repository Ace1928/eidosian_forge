import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def _get_alpha_y(a, n, k):
    """Computes the rotation angles required to implement the uniformly controlled Y rotation
    applied to the :math:`k`th qubit.

    The :math:`j`-th angle is related to the absolute values, a, of the desired amplitudes via:

    .. math:: \\alpha^{y,k}_j = 2 \\arcsin \\sqrt{ \\frac{ \\sum_{l=1}^{2^{k-1}} a_{(2j-1)2^{k-1} +l}^2  }{ \\sum_{l=1}^{2^{k}} a_{(j-1)2^{k} +l}^2  } }

    Args:
        a (tensor_like): absolute values of the state to prepare
        n (int): total number of qubits for the uniformly-controlled rotation
        k (int): index of current qubit

    Returns:
        array representing :math:`\\alpha^{y,k}`
    """
    indices_numerator = [[(2 * (j + 1) - 1) * 2 ** (k - 1) + l for l in range(2 ** (k - 1))] for j in range(2 ** (n - k))]
    numerator = qml.math.take(a, indices=indices_numerator, axis=-1)
    numerator = qml.math.sum(qml.math.abs(numerator) ** 2, axis=-1)
    indices_denominator = [[j * 2 ** k + l for l in range(2 ** k)] for j in range(2 ** (n - k))]
    denominator = qml.math.take(a, indices=indices_denominator, axis=-1)
    denominator = qml.math.sum(qml.math.abs(denominator) ** 2, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        division = numerator / denominator
    division = qml.math.cast(division, np.float64)
    denominator = qml.math.cast(denominator, np.float64)
    division = qml.math.where(denominator != 0.0, division, 0.0)
    return 2 * qml.math.arcsin(qml.math.sqrt(division))