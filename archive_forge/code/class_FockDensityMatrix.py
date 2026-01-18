import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class FockDensityMatrix(CVOperation):
    """
    Prepare subsystems using the given density matrix in the Fock basis.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        state (array): a single mode matrix :math:`\\rho_{ij}`, or
            a multimode tensor :math:`\\rho_{ij,kl,\\dots,mn}`, with two indices per mode
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = AnyWires
    grad_method = 'F'

    def __init__(self, state, wires, id=None):
        super().__init__(state, wires=wires, id=id)