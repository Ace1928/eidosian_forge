from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _bisect_compute_b(u: np.ndarray):
    """
    Given the U matrix, compute the B matrix such that
    H Bt x B x H = U
    where Bt is the adjoint of B,
    H is the Hadamard matrix,
    and x is the Pauli X matrix.
    """
    sqrt = np.sqrt
    Abs = np.abs
    w = np.real(u[0, 0])
    s = np.real(u[1, 0])
    t = np.imag(u[1, 0])
    if np.isclose(s, 0):
        b = 0
        if np.isclose(t, 0):
            if w < 0:
                c = 0
                d = sqrt(-w)
            else:
                c = sqrt(w)
                d = 0
        else:
            c = sqrt(2 - 2 * w) * (-w / 2 - 1 / 2) / t
            d = sqrt(2 - 2 * w) / 2
    elif np.isclose(t, 0):
        b = (1 / 2 - w / 2) * sqrt(2 * w + 2) / s
        c = sqrt(2 * w + 2) / 2
        d = 0
    else:
        b = sqrt(2) * s * sqrt((1 - w) / (s ** 2 + t ** 2)) * Abs(t) / (2 * t)
        c = sqrt(2) * sqrt((1 - w) / (s ** 2 + t ** 2)) * (w + 1) * Abs(t) / (2 * t)
        d = -sqrt(2) * sqrt((1 - w) / (s ** 2 + t ** 2)) * Abs(t) / 2
    return _param_su2(c, d, b, 0)