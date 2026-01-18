import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class GaussianState(CVOperation):
    """
    Prepare subsystems in a given Gaussian state.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 2
    * Gradient recipe: None

    Args:
        V (array): the :math:`2N\\times 2N` (real and positive definite) covariance matrix
        r (array): a length :math:`2N` vector of means, of the
            form :math:`(\\x_0,\\dots,\\x_{N-1},\\p_0,\\dots,\\p_{N-1})`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 2
    num_wires = AnyWires
    grad_method = 'F'

    def __init__(self, V, r, wires, id=None):
        super().__init__(V, r, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'Gaussian', cache=cache)