import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class InterferometerUnitary(CVOperation):
    """
    A linear interferometer transforming the bosonic operators according to
    the unitary matrix :math:`U`.

    .. note::

        This operation implements a **fixed** linear interferometer given a known
        unitary matrix.

        If you instead wish to parameterize the interferometer,
        and calculate the gradient/optimize with respect to these parameters,
        consider instead the :func:`pennylane.template.Interferometer` template,
        which constructs an interferometer from a combination of beamsplitters
        and rotation gates.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None
    * Heisenberg representation:

      .. math:: M = \\begin{bmatrix}
        1 & 0\\\\
        0 & S\\\\
        \\end{bmatrix}

    where :math:`S` is the Gaussian symplectic transformation representing the interferometer.

    Args:
        U (array): A shape ``(len(wires), len(wires))`` complex unitary matrix
        wires (Sequence[Any] or Any): the wires the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 1
    num_wires = AnyWires
    grad_method = None
    grad_recipe = None

    def __init__(self, U, wires, id=None):
        super().__init__(U, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        N = len(p[0])
        A = p[0].real
        B = p[0].imag
        rows = np.arange(2 * N).reshape(2, -1).T.flatten()
        S = np.vstack([np.hstack([A, -B]), np.hstack([B, A])])[:, rows][rows]
        M = np.eye(2 * N + 1)
        M[1:2 * N + 1, 1:2 * N + 1] = S
        return M

    def adjoint(self):
        U = self.parameters[0]
        return InterferometerUnitary(qml_math.T(qml_math.conj(U)), wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'U', cache=cache)