import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class ControlledAddition(CVOperation):
    """
    Controlled addition operation.

    .. math::
           \\text{CX}(s) = \\int dx \\ket{x}\\bra{x} \\otimes D\\left({\\frac{1}{\\sqrt{2\\hbar}}}s x\\right)
           = e^{-i s \\: \\hat{x} \\otimes \\hat{p}/\\hbar}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\\frac{d}{ds}f(\\text{CX}(s)) = \\frac{1}{2 a} \\left[f(\\text{CX}(s+a)) - f(\\text{CX}(s-a))\\right]`,
      where :math:`a` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`\\text{CX}(s)`.

    * Heisenberg representation:

      .. math:: M = \\begin{bmatrix}
            1 & 0 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 & -s \\\\
            0 & s & 0 & 1 & 0 \\\\
            0 & 0 & 0 & 0 & 1
        \\end{bmatrix}

    Args:
        s (float): addition multiplier
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
        U[2, 4] = -p[0]
        U[3, 1] = p[0]
        return U

    def adjoint(self):
        return ControlledAddition(-self.parameters[0], wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'X', cache=cache)