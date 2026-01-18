from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _ctrl_decomp_bisect_md(u: np.ndarray, target_wire: Wires, control_wires: Wires):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1, Theorem 2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real main-diagonal for this specialized method to work.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        u (np.ndarray): the target operation's matrix
        target_wire (~.wires.Wires): the target wire of the controlled operation
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``u`` does not have a real main-diagonal

    """
    ui = np.imag(u)
    if not np.isclose(ui[0, 0], 0) or not np.isclose(ui[1, 1], 0):
        raise ValueError(f"Target operation's matrix must have real main-diagonal, but it is {u}")
    h_matrix = qml.Hadamard.compute_matrix()
    mod_u = h_matrix @ u @ h_matrix
    decomposition = [qml.Hadamard(target_wire)]
    decomposition += _ctrl_decomp_bisect_od(mod_u, target_wire, control_wires)
    decomposition.append(qml.Hadamard(target_wire))
    return decomposition