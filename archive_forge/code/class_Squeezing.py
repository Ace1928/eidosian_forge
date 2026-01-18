import math
import numpy as np
from scipy.linalg import block_diag
from pennylane import math as qml_math
from pennylane.operation import AnyWires, CVObservable, CVOperation
from .identity import Identity, I  # pylint: disable=unused-import
from .meta import Snapshot  # pylint: disable=unused-import
class Squeezing(CVOperation):
    """
    Phase space squeezing.

    .. math::
        S(z) = \\exp\\left(\\frac{1}{2}(z^* \\a^2 -z {\\a^\\dagger}^2)\\right).

    where :math:`z = r e^{i\\phi}`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: :math:`\\frac{d}{dr}f(S(r,\\phi)) = \\frac{1}{2\\sinh s} \\left[f(S(r+s, \\phi)) - f(S(r-s, \\phi))\\right]`,
      where :math:`s` is an arbitrary real number (:math:`0.1` by default) and
      :math:`f` is an expectation value depending on :math:`S(r,\\phi)`.
    * Heisenberg representation:

      .. math:: M = \\begin{bmatrix}
        1 & 0 & 0 \\\\
        0 & \\cosh r - \\cos\\phi \\sinh r & -\\sin\\phi\\sinh r \\\\
        0 & -\\sin\\phi\\sinh r & \\cosh r+\\cos\\phi\\sinh r
        \\end{bmatrix}

    Args:
        r (float): squeezing amount
        phi (float): squeezing phase angle :math:`\\phi`
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_params = 2
    num_wires = 1
    grad_method = 'A'
    shift = 0.1
    multiplier = 0.5 / math.sinh(shift)
    a = 1
    grad_recipe = ([[multiplier, a, shift], [-multiplier, a, -shift]], _two_term_shift_rule)

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1] / 2)
        return R @ np.diag([1, math.exp(-p[0]), math.exp(p[0])]) @ R.T

    def adjoint(self):
        r, phi = self.parameters
        new_phi = (phi + np.pi) % (2 * np.pi)
        return Squeezing(r, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'S', cache=cache)