from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _ctrl_decomp_bisect_od(u: np.ndarray, target_wire: Wires, control_wires: Wires):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1, Theorem 1 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real off-diagonal for this specialized method to work.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        u (np.ndarray): the target operation's matrix
        target_wire (~.wires.Wires): the target wire of the controlled operation
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``u`` does not have a real off-diagonal

    """
    ui = np.imag(u)
    if not np.isclose(ui[1, 0], 0) or not np.isclose(ui[0, 1], 0):
        raise ValueError(f"Target operation's matrix must have real off-diagonal, but it is {u}")
    a = _bisect_compute_a(u)
    mid = (len(control_wires) + 1) // 2
    control_k1 = control_wires[:mid]
    control_k2 = control_wires[mid:]

    def component():
        return [qml.ctrl(qml.X(target_wire), control=control_k1, work_wires=control_k2), qml.QubitUnitary(a, target_wire), qml.ctrl(qml.X(target_wire), control=control_k2, work_wires=control_k1), qml.adjoint(qml.QubitUnitary(a, target_wire))]
    return component() + component()