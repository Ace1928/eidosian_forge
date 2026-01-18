import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class NumberOperator(CVObservable):
    """
    The photon number observable :math:`\\langle \\hat{n}\\rangle`.

    The number operator is defined as
    :math:`\\hat{n} = \\a^\\dagger \\a = \\frac{1}{2\\hbar}(\\x^2 +\\p^2) -\\I/2`.

    When used with the :func:`~pennylane.expval` function, the mean
    photon number :math:`\\braket{\\hat{n}}` is returned.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Observable order: 2nd order in the quadrature operators
    * Heisenberg representation:

      .. math:: M = \\frac{1}{2\\hbar}\\begin{bmatrix}
            -\\hbar & 0 & 0\\\\
            0 & 1 & 0\\\\
            0 & 0 & 1
        \\end{bmatrix}

    Args:
        wires (Sequence[Any] or Any): the wire the operation acts on
    """
    num_params = 0
    num_wires = 1
    ev_order = 2

    def __init__(self, wires):
        super().__init__(wires=wires)

    @staticmethod
    def _heisenberg_rep(p):
        hbar = 2
        return np.diag([-0.5, 0.5 / hbar, 0.5 / hbar])

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or 'n'