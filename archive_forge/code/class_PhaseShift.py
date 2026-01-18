import functools
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
class PhaseShift(Operation):
    """
    Arbitrary single qubit local phase shift

    .. math:: R_\\phi(\\phi) = e^{i\\phi/2}R_z(\\phi) = \\begin{bmatrix}
                1 & 0 \\\\
                0 & e^{i\\phi}
            \\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(R_\\phi(\\phi)) = \\frac{1}{2}\\left[f(R_\\phi(\\phi+\\pi/2)) - f(R_\\phi(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`R_{\\phi}(\\phi)`.

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
        return qml.Projector(np.array([1]), wires=self.wires)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'Rϕ', cache=cache)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PhaseShift.matrix`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.PhaseShift.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.9689+0.2474j]])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            p = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            ones = qml.math.ones_like(p)
            zeros = qml.math.zeros_like(p)
            return qml.math.stack([stack_last([ones, zeros]), stack_last([zeros, p])], axis=-2)
        signs = qml.math.array([0, 1], like=phi)
        arg = 1j * phi
        if qml.math.ndim(arg) == 0:
            return qml.math.diag(qml.math.exp(arg * signs))
        diags = qml.math.exp(qml.math.outer(arg, signs))
        return diags[:, :, np.newaxis] * qml.math.cast_like(qml.math.eye(2, like=diags), diags)

    @staticmethod
    def compute_eigvals(phi):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PhaseShift.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.PhaseShift.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 0.8776+0.4794j])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
            return stack_last([qml.math.ones_like(phase), phase])
        prefactors = qml.math.array([0, 1j], like=phi)
        if qml.math.ndim(phi) == 0:
            product = phi * prefactors
        else:
            product = qml.math.outer(phi, prefactors)
        return qml.math.exp(product)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.PhaseShift.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Any, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PhaseShift.compute_decomposition(1.234, wires=0)
        [RZ(1.234, wires=[0]), GlobalPhase(-0.617, wires=[])]

        """
        return [RZ(phi, wires=wires), qml.GlobalPhase(-phi / 2)]

    def adjoint(self):
        return PhaseShift(-self.data[0], wires=self.wires)

    def pow(self, z):
        return [PhaseShift(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        return qml.ControlledPhaseShift(*self.parameters, wires=wire + self.wires)

    def simplify(self):
        phi = self.data[0] % (2 * np.pi)
        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires)
        return PhaseShift(phi, wires=self.wires)

    def single_qubit_rot_angles(self):
        return [self.data[0], 0.0, 0.0]