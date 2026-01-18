import copy
import numpy as np
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.ops import BlockEncode, PCPhase
from pennylane.ops.op_math import adjoint
from pennylane.operation import AnyWires, Operation
def _qsp_to_qsvt(angles):
    """Converts qsp angles to qsvt angles."""
    new_angles = qml.math.array(copy.copy(angles))
    new_angles[0] += 3 * np.pi / 4
    new_angles[-1] -= np.pi / 4
    new_angles[1:-1] += np.pi / 2
    return new_angles