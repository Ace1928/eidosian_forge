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
class PSWAP(Operation):
    """Phase SWAP gate

    .. math:: PSWAP(\\phi) = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 0 & e^{i \\phi} & 0 \\\\
            0 & e^{i \\phi} & 0 & 0 \\\\
            0 & 0 & 0 & 1
        \\end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe:

    .. math::
        \\frac{d}{d \\phi} PSWAP(\\phi)
        = \\frac{1}{2} \\left[ PSWAP(\\phi + \\pi / 2) - PSWAP(\\phi - \\pi / 2) \\right]

    Args:
        phi (float): the phase angle
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    grad_method = 'A'
    grad_recipe = ([[0.5, 1, np.pi / 2], [-0.5, 1, -np.pi / 2]],)

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.PSWAP.decomposition`.

        Args:
            phi (float): the phase angle
            wires (Iterable, Wires): the subsystem the gate acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.PSWAP.compute_decomposition(1.23, wires=(0,1))
        [SWAP(wires=[0, 1]), CNOT(wires=[0, 1]), PhaseShift(1.23, wires=[1]), CNOT(wires=[0, 1])]
        """
        return [qml.SWAP(wires=wires), qml.CNOT(wires=wires), PhaseShift(phi, wires=[wires[1]]), qml.CNOT(wires=wires)]

    @staticmethod
    def compute_matrix(phi):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PSWAP.matrix`


        Args:
           phi (tensor_like or float): phase angle

        Returns:
           tensor_like: canonical matrix

        **Example**

        >>> qml.PSWAP.compute_matrix(0.5)
        array([[1.        +0.j, 0.        +0.j        , 0.        +0.j        , 0.        +0.j],
              [0.        +0.j, 0.        +0.j        , 0.87758256+0.47942554j, 0.        +0.j],
              [0.        +0.j, 0.87758256+0.47942554j, 0.        +0.j        , 0.        +0.j],
              [0.        +0.j, 0.        +0.j        , 0.        +0.j        , 1.        +0.j]])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phi = qml.math.cast_like(phi, 1j)
        e = qml.math.exp(1j * phi)
        return qml.math.stack([stack_last([1, 0, 0, 0]), stack_last([0, 0, e, 0]), stack_last([0, e, 0, 0]), stack_last([0, 0, 0, 1])], axis=-2)

    @staticmethod
    def compute_eigvals(phi):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PSWAP.eigvals`


        Args:
            phi (tensor_like or float): phase angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.PSWAP.compute_eigvals(0.5)
        array([ 1.        +0.j        ,  1.        +0.j,       -0.87758256-0.47942554j,  0.87758256+0.47942554j])
        """
        if qml.math.get_interface(phi) == 'tensorflow':
            phi = qml.math.cast_like(phi, 1j)
        return qml.math.stack([1, 1, -qml.math.exp(1j * phi), qml.math.exp(1j * phi)])

    def adjoint(self):
        phi, = self.parameters
        return PSWAP(-phi, wires=self.wires)

    def simplify(self):
        phi = self.data[0] % (2 * np.pi)
        if _can_replace(phi, 0):
            return qml.SWAP(wires=self.wires)
        return PSWAP(phi, wires=self.wires)