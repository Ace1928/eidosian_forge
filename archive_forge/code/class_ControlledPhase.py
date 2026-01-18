import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class ControlledPhase(CVOperation):
    """
    Controlled phase operation.

    .. math::
           \\text{CZ}(s) =  \\iint dx dy \\: e^{i sxy/\\hbar} \\ket{x,y}\\bra{x,y}
           = e^{i s \\: \\hat{x} \\otimes \\hat{x}/\\hbar}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\\frac{d}{ds}f(\\text{CZ}(s)) = \\frac{1}{2 a} \\left[f(\\text{CZ}(s+a)) - f(\\text{CZ}(s-a))\\right]`,
      where :math:`a` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`\\text{CZ}(s)`.

    * Heisenberg representation:

      .. math:: M = \\begin{bmatrix}
            1 & 0 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & s & 0 \\\\
            0 & 0 & 0 & 1 & 0 \\\\
            0 & s & 0 & 0 & 1
        \\end{bmatrix}

    Args:
        s (float):  phase shift multiplier
        wires (Sequence[Any]): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 2
    grad_method = 'A'
    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]],)

    def __init__(self, s, wires, id=None):
        super().__init__(s, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(5)
        U[2, 3] = p[0]
        U[4, 1] = p[0]
        return U

    def adjoint(self):
        return ControlledPhase(-self.parameters[0], wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'Z', cache=cache)