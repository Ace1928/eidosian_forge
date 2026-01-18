from quantum chemistry applications.
import functools
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import Operation
class SingleExcitationMinus(Operation):
    """
    Single excitation rotation with negative phase-shift outside the rotation subspace.

    .. math:: U_-(\\phi) = \\begin{bmatrix}
                e^{-i\\phi/2} & 0 & 0 & 0 \\\\
                0 & \\cos(\\phi/2) & -\\sin(\\phi/2) & 0 \\\\
                0 & \\sin(\\phi/2) & \\cos(\\phi/2) & 0 \\\\
                0 & 0 & 0 & e^{-i\\phi/2}
            \\end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(U_-(\\phi)) = \\frac{1}{2}\\left[f(U_-(\\phi+\\pi/2)) - f(U_-(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`U_-(\\phi)`.

    Args:
        phi (float): rotation angle :math:`\\phi`
        wires (Sequence[int] or int): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    """
    num_wires = 2
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
        w1, w2 = self.wires
        return 0.25 * (-qml.Identity(w1) + qml.X(w1) @ qml.Y(w2) - qml.Y(w1) @ qml.X(w2) - qml.Z(w1) @ qml.Z(w2))

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SingleExcitationMinus.matrix`


        Args:
          phi (tensor_like or float): rotation angle

        Returns:
          tensor_like: canonical matrix

        **Example**

        >>> qml.SingleExcitationMinus.compute_matrix(torch.tensor(0.5))
        tensor([[ 0.9689-0.2474j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.9689+0.0000j, -0.2474+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.2474+0.0000j,  0.9689+0.0000j,  0.0000+0.0000j],
                [ 0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,  0.9689-0.2474j]])
        """
        return _single_excitations_matrix(phi, -0.5j)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.SingleExcitationMinus.decomposition`.

        Args:
            phi (float): rotation angle :math:`\\phi`
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.SingleExcitationMinus.compute_decomposition(1.23, wires=(0,1))
        [X(0),
        X(1),
        ControlledPhaseShift(-0.615, wires=[1, 0]),
        X(0),
        X(1),
        ControlledPhaseShift(-0.615, wires=[0, 1]),
        CNOT(wires=[0, 1]),
        CRY(1.23, wires=[1, 0]),
        CNOT(wires=[0, 1])]

        """
        decomp_ops = [qml.X(wires[0]), qml.X(wires[1]), qml.ControlledPhaseShift(-phi / 2, wires=[wires[1], wires[0]]), qml.X(wires[0]), qml.X(wires[1]), qml.ControlledPhaseShift(-phi / 2, wires=[wires[0], wires[1]]), qml.CNOT(wires=[wires[0], wires[1]]), qml.CRY(phi, wires=[wires[1], wires[0]]), qml.CNOT(wires=[wires[0], wires[1]])]
        return decomp_ops

    def adjoint(self):
        phi, = self.parameters
        return SingleExcitationMinus(-phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'G₋', cache=cache)