import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
class RZ(Operation):
    """
    The single qubit Z rotation

    .. math:: R_z(\\phi) = e^{-i\\phi\\sigma_z/2} = \\begin{bmatrix}
                e^{-i\\phi/2} & 0 \\\\
                0 & e^{i\\phi/2}
            \\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(R_z(\\phi)) = \\frac{1}{2}\\left[f(R_z(\\phi+\\pi/2)) - f(R_z(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`R_z(\\phi)`.

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
    basis = 'Z'
    grad_method = 'A'
    parameter_frequencies = [(1,)]

    def generator(self):
        return -0.5 * PauliZ(wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(theta):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.RZ.matrix`

        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.RZ.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(theta) == 'tensorflow':
            p = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            z = qml.math.zeros_like(p)
            return qml.math.stack([stack_last([p, z]), stack_last([z, qml.math.conj(p)])], axis=-2)
        signs = qml.math.array([-1, 1], like=theta)
        arg = 0.5j * theta
        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))
        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(2, like=diags), diags)

    @staticmethod
    def compute_eigvals(theta):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.RZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.RZ.compute_eigvals(torch.tensor(0.5))
        tensor([0.9689-0.2474j, 0.9689+0.2474j])
        """
        if qml.math.get_interface(theta) == 'tensorflow':
            phase = qml.math.exp(-0.5j * qml.math.cast_like(theta, 1j))
            return qml.math.stack([phase, qml.math.conj(phase)], axis=-1)
        prefactors = qml.math.array([-0.5j, 0.5j], like=theta)
        if qml.math.ndim(theta) == 0:
            product = theta * prefactors
        else:
            product = qml.math.outer(theta, prefactors)
        return qml.math.exp(product)

    def adjoint(self):
        return RZ(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [RZ(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CRZ(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        theta = self.data[0] % (4 * np.pi)
        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)
        return RZ(theta, wires=self.wires)

    def single_qubit_rot_angles(self):
        return [self.data[0], 0.0, 0.0]