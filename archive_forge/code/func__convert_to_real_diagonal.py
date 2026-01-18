from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _convert_to_real_diagonal(q: np.ndarray) -> np.ndarray:
    """
    Change the phases of Q so the main diagonal is real, and return the modified Q.
    """
    exp_angles = np.angle(np.diag(q))
    return q * np.exp(-1j * exp_angles).reshape((1, 2))