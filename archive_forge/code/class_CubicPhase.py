import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class CubicPhase(CVOperation):
    """
    Cubic phase shift.

    .. math::
        V(\\gamma) = e^{i \\frac{\\gamma}{3} \\hat{x}^3/\\hbar}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        gamma (float): parameter
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = 'F'

    def __init__(self, gamma, wires, id=None):
        super().__init__(gamma, wires=wires, id=id)

    def adjoint(self):
        return CubicPhase(-self.parameters[0], wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'V', cache=cache)