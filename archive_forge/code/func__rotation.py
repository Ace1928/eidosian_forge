import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
def _rotation(phi, bare=False):
    """Utility function, returns the Heisenberg transformation of a phase rotation gate.

    The transformation matrix returned is:

    .. math:: M = \\begin{bmatrix}
        1 & 0 & 0\\\\
        0 & \\cos\\phi & -\\sin\\phi\\\\
        0 & \\sin\\phi & \\cos\\phi
        \\end{bmatrix}

    Args:
        phi (float): rotation angle.
        bare (bool): if True, return a simple 2d rotation matrix

    Returns:
        array[float]: transformation matrix
    """
    c = math.cos(phi)
    s = math.sin(phi)
    temp = np.array([[c, -s], [s, c]])
    if bare:
        return temp
    return block_diag(1, temp)