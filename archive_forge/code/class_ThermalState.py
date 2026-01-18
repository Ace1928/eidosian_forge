import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class ThermalState(CVOperation):
    """
    Prepares a thermal state.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        nbar (float): mean thermal population of the mode
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = 'F'

    def __init__(self, nbar, wires, id=None):
        super().__init__(nbar, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'Thermal', cache=cache)