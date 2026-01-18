import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class CrossKerr(CVOperation):
    """
    Cross-Kerr interaction.

    .. math::
        CK(\\kappa) = e^{i \\kappa \\hat{n}_1\\hat{n}_2}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        kappa (float): parameter
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 2
    grad_method = 'F'

    def __init__(self, kappa, wires, id=None):
        super().__init__(kappa, wires=wires, id=id)

    def adjoint(self):
        return CrossKerr(-self.parameters[0], wires=self.wires)