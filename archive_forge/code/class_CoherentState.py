import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class CoherentState(CVOperation):
    """
    Prepares a coherent state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: None (uses finite difference)

    Args:
        a (float): displacement magnitude :math:`r=|\\alpha|`
        phi (float): phase angle :math:`\\phi`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 2
    num_wires = 1
    grad_method = 'F'

    def __init__(self, a, phi, wires, id=None):
        super().__init__(a, phi, wires=wires, id=id)