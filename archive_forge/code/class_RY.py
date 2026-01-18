import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
class RY(Operation):
    """
    The single qubit Y rotation

    .. math:: R_y(\\phi) = e^{-i\\phi\\sigma_y/2} = \\begin{bmatrix}
                \\cos(\\phi/2) & -\\sin(\\phi/2) \\\\
                \\sin(\\phi/2) & \\cos(\\phi/2)
            \\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(R_y(\\phi)) = \\frac{1}{2}\\left[f(R_y(\\phi+\\pi/2)) - f(R_y(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`R_y(\\phi)`.

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int] or int): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    basis = 'Y'
    grad_method = 'A'
    parameter_frequencies = [(1,)]

    def generator(self):
        return -0.5 * PauliY(wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RY.matrix`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RY.compute_matrix(torch.tensor(0.5))
        tensor([[ 0.9689, -0.2474],
                [ 0.2474,  0.9689]])
        """
        c = qml.math.cos(theta / 2)
        s = qml.math.sin(theta / 2)
        if qml.math.get_interface(theta) == 'tensorflow':
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
        c = (1 + 0j) * c
        s = (1 + 0j) * s
        return qml.math.stack([stack_last([c, -s]), stack_last([s, c])], axis=-2)

    def adjoint(self):
        return RY(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RY(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRY(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)
        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)
        return RY(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        return [0.0, self.data[0], 0.0]