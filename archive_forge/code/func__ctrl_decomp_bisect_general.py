from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _ctrl_decomp_bisect_general(u: np.ndarray, target_wire: Wires, control_wires: Wires):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        u (np.ndarray): the target operation's matrix
        target_wire (~.wires.Wires): the target wire of the controlled operation
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations
    """
    x_matrix = qml.X.compute_matrix()
    h_matrix = qml.Hadamard.compute_matrix()
    alternate_h_matrix = x_matrix @ h_matrix @ x_matrix
    d, q = npl.eig(u)
    d = np.diag(d)
    q = _convert_to_real_diagonal(q)
    b = _bisect_compute_b(q)
    c1 = b @ alternate_h_matrix
    c2t = b @ h_matrix
    mid = (len(control_wires) + 1) // 2
    control_k1 = control_wires[:mid]
    control_k2 = control_wires[mid:]
    component = [qml.QubitUnitary(c2t, target_wire), qml.ctrl(qml.X(target_wire), control=control_k2, work_wires=control_k1), qml.adjoint(qml.QubitUnitary(c1, target_wire)), qml.ctrl(qml.X(target_wire), control=control_k1, work_wires=control_k2)]
    od_decomp = _ctrl_decomp_bisect_od(d, target_wire, control_wires)
    qml.QueuingManager.remove(component[3])
    qml.QueuingManager.remove(od_decomp[0])
    adjoint_component = [qml.adjoint(copy(op), lazy=False) for op in reversed(component)]
    return component[0:3] + od_decomp[1:] + adjoint_component