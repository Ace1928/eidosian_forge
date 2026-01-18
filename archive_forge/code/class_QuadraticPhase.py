import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class QuadraticPhase(CVOperation):
    """
    Quadratic phase shift.

    .. math::
        P(s) = e^{i \\frac{s}{2} \\hat{x}^2/\\hbar}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\\frac{d}{ds}f(P(s)) = \\frac{1}{2 a} \\left[f(P(s+a)) - f(P(s-a))\\right]`,
      where :math:`a` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`P(s)`.

    * Heisenberg representation:

      .. math:: M = \\begin{bmatrix}
            1 & 0 & 0 \\\\
            0 & 1 & 0 \\\\
            0 & s & 1 \\\\
        \\end{bmatrix}

    Args:
        s (float): parameter
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = 1
    grad_method = 'A'
    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]],)

    def __init__(self, s, wires, id=None):
        super().__init__(s, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        U = np.identity(3)
        U[2, 1] = p[0]
        return U

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'P', cache=cache)