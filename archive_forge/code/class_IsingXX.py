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
class IsingXX(Operation):
    """
    Ising XX coupling gate

    .. math:: XX(\\phi) = \\exp(-i \\frac{\\phi}{2} (X \\otimes X)) =
        \\begin{bmatrix} =
            \\cos(\\phi / 2) & 0 & 0 & -i \\sin(\\phi / 2) \\\\
            0 & \\cos(\\phi / 2) & -i \\sin(\\phi / 2) & 0 \\\\
            0 & -i \\sin(\\phi / 2) & \\cos(\\phi / 2) & 0 \\\\
            -i \\sin(\\phi / 2) & 0 & 0 & \\cos(\\phi / 2)
        \\end{bmatrix}.

    .. note::

        Special cases of using the :math:`XX` operator include:

        * :math:`XX(0) = I`;
        * :math:`XX(\\pi) = i (X \\otimes X)`.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)
    * Gradient recipe: :math:`\\frac{d}{d\\phi}f(XX(\\phi)) = \\frac{1}{2}\\left[f(XX(\\phi +\\pi/2)) - f(XX(\\phi-\\pi/2))\\right]`
      where :math:`f` is an expectation value depending on :math:`XX(\\phi)`.

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
        return -0.5 * PauliX(wires=self.wires[0]) @ PauliX(wires=self.wires[1])

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.

        .. seealso:: :meth:`~.IsingXX.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.IsingXX.compute_matrix(torch.tensor(0.5))
        tensor([[0.9689+0.0000j, 0.0000+0.0000j, 0.0000+0.0000j, 0.0000-0.2474j],
                [0.0000+0.0000j, 0.9689+0.0000j, 0.0000-0.2474j, 0.0000+0.0000j],
                [0.0000+0.0000j, 0.0000-0.2474j, 0.9689+0.0000j, 0.0000+0.0000j],
                [0.0000-0.2474j, 0.0000+0.0000j, 0.0000+0.0000j, 0.9689+0.0000j]],
               dtype=torch.complex128)
        """
        c = qml.math.cos(phi / 2)
        s = qml.math.sin(phi / 2)
        eye = qml.math.eye(4, like=phi)
        rev_eye = qml.math.convert_like(np.eye(4)[::-1].copy(), phi)
        if qml.math.get_interface(phi) == 'tensorflow':
            c = qml.math.cast_like(c, 1j)
            s = qml.math.cast_like(s, 1j)
            eye = qml.math.cast_like(eye, 1j)
            rev_eye = qml.math.cast_like(rev_eye, 1j)
        js = -1j * s
        if qml.math.ndim(phi) == 0:
            return c * eye + js * rev_eye
        return qml.math.tensordot(c, eye, axes=0) + qml.math.tensordot(js, rev_eye, axes=0)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.IsingXX.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.IsingXX.compute_decomposition(1.23, wires=(0,1))
        [CNOT(wires=[0, 1]), RX(1.23, wires=[0]), CNOT(wires=[0, 1]]

        """
        decomp_ops = [qml.CNOT(wires=wires), RX(phi, wires=[wires[0]]), qml.CNOT(wires=wires)]
        return decomp_ops

    def adjoint(self):
        phi, = self.parameters
        return IsingXX(-phi, wires=self.wires)

    def pow(self, z):
        return [IsingXX(self.data[0] * z, wires=self.wires)]

    def simplify(self):
        phi = self.data[0] % (4 * np.pi)
        if _can_replace(phi, 0):
            return qml.Identity(wires=self.wires[0])
        return IsingXX(phi, wires=self.wires)