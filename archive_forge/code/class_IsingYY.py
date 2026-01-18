import functools
from operator import matmul
import numpy as np
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.operation import AnyWires, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
from .non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from .parametric_ops_single_qubit import _can_replace, stack_last, RX, RY, RZ, PhaseShift
class IsingYY(Operation):
    """
    Ising YY coupling gate

    .. math:: \\mathtt{YY}(\\phi) = \\exp(-i \\frac{\\phi}{2} (Y \\otimes Y)) =
        \\begin{bmatrix}
            \\cos(\\phi / 2) & 0 & 0 & i \\sin(\\phi / 2) \\\\
            0 & \\cos(\\phi / 2) & -i \\sin(\\phi / 2) & 0 \\\\
            0 & -i \\sin(\\phi / 2) & \\cos(\\phi / 2) & 0 \\\\
            i \\sin(\\phi / 2) & 0 & 0 & \\cos(\\phi / 2)
        \\end{bmatrix}.

    .. note::

        Special cases of using the :math:`YY` operator include:

        * :math:`YY(0) = I`;
        * :math:`YY(\\pi) = i (Y \\otimes Y)`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(YY(\\phi)) = \\frac{1}{2}\\left[f(YY(\\phi +\\pi/2)) - f(YY(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`YY(\\phi)`.

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (0,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = 'A'
    parameter_frequencies = [(1,)]

    def generator(self):
        return -0.5 * PauliY(wires=self.wires[0]) @ PauliY(wires=self.wires[1])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.IsingYY.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingYY.compute_decomposition(1.23, wires=(0,1))
        [CY(wires=[0, 1]), RY(1.23, wires=[0]), CY(wires=[0, 1])]

        """
        return [qml.CY(wires=wires), RY(phi, wires=[wires[0]]), qml.CY(wires=wires)]

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IsingYY.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingYY.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000+0.2474j],
                [0.0000+0.0000j, 0.9689+0.0000j, 0.0000-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000-0.2474j, 0.9689+0.0000j, 0.0000+0.0000j],
                [0.0000+0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.0000j]])
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        if qml.math.get_interface(phi) == 'tensorflow':
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
        js = 1j * s
        r_term = qml.math.cast_like(qml.math.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], like=js), 1j)
        if qml.math.ndim(phi) == 0:
            return c * qml.math.cast_like(qml.math.eye(4, like=c), c) + js * r_term
        return qml.math.tensordot(c, np.eye(4), axes=0) + qml.math.tensordot(js, r_term, axes=0)

    def adjoint(self):
        phi, = self.parameters
        return IsingYY(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingYY(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)
        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])
        return IsingYY(phi, wires=self.wires)