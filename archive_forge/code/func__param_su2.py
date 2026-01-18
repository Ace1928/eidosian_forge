from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _param_su2(ar: float, ai: float, br: float, bi: float):
    """
    Create a matrix in the SU(2) form from complex parameters a, b.
    The resulting matrix is not guaranteed to be in SU(2), unless |a|^2 + |b|^2 = 1.
    """
    return np.array([[ar + 1j * ai, -br + 1j * bi], [br + 1j * bi, ar + 1j * -ai]])