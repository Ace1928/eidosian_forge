from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
class DoubleExcitationPlus(Operation):
    """
    Double excitation rotation with positive phase-shift outside the rotation subspace.

    This operation performs an :math:`SO(2)` rotation in the two-dimensional subspace :math:`\\{
    |1100\\rangle,|0011\\rangle\\}` while applying a phase-shift on other states. More precisely,
    it performs the transformation

    .. math::

        &|0011\\rangle \\rightarrow \\cos(\\phi/2) |0011\\rangle - \\sin(\\phi/2) |1100\\rangle\\\\
        &|1100\\rangle \\rightarrow \\cos(\\phi/2) |1100\\rangle + \\sin(\\phi/2) |0011\\rangle\\\\
        &|x\\rangle \\rightarrow e^{i\\phi/2} |x\\rangle,

    for all other basis states :math:`|x\\rangle`.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(U_+(\\phi)) = \\frac{1}{2}\\left[f(U_+(\\phi+\\pi/2)) - f(U_+(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`U_+(\\phi)`

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 4
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    'Gradient computation method.'
    parameter_frequencies = [(1,)]
    'Frequencies of the operation parameter with respect to an expectation value.'

    def generator(self):
        G = -1 * np.eye(16, dtype=np.complex64)
        G[3, 3] = G[12, 12] = 0
        G[3, 12] = -1j
        G[12, 3] = 1j
        H = csr_matrix(-0.5 * G)
        return qml.SparseHamiltonian(H, wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DoubleExcitationPlus.matrix`

        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        """
        return _double_excitations_matrix(phi, 0.5j)

    def adjoint(self):
        theta, = self.parameters
        return DoubleExcitationPlus(-theta, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'G²₊', cache=cache)