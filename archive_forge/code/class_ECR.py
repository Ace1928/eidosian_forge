import cmath
from copy import copy
from functools import lru_cache
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane.operation import Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
class ECR(Operation):
    """ ECR(wires)

    An echoed RZX(pi/2) gate.

    .. math:: ECR = {1/\\sqrt{2}} \\begin{bmatrix}
            0 & 0 & 1 & i \\\\
            0 & 0 & i & 1 \\\\
            1 & -i & 0 & 0 \\\\
            -i & 1 & 0 & 0
        \\end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 2
    num_params = 0
    batch_size = None

    @staticmethod
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ECR.matrix`


        Return type: tensor_like

        **Example**

        >>> print(qml.ECR.compute_matrix())
         [[0+0.j 0.+0.j 1/sqrt(2)+0.j 0.+1j/sqrt(2)]
         [0.+0.j 0.+0.j 0.+1.j/sqrt(2) 1/sqrt(2)+0.j]
         [1/sqrt(2)+0.j 0.-1.j/sqrt(2) 0.+0.j 0.+0.j]
         [0.-1/sqrt(2)j 1/sqrt(2)+0.j 0.+0.j 0.+0.j]]
        """
        return np.array([[0, 0, INV_SQRT2, INV_SQRT2 * 1j], [0, 0, INV_SQRT2 * 1j, INV_SQRT2], [INV_SQRT2, -INV_SQRT2 * 1j, 0, 0], [-INV_SQRT2 * 1j, INV_SQRT2, 0, 0]])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ECR.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.ECR.compute_eigvals())
        [1, -1, 1, -1]
        """
        return np.array([1, -1, 1, -1])

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

           .. math:: O = O_1 O_2 \\dots O_n.


           .. seealso:: :meth:`~.ECR.decomposition`.

           Args:
               wires (Iterable, Wires): wires that the operator acts on

           Returns:
               list[Operator]: decomposition into lower level operations

           **Example:**

           >>> print(qml.ECR.compute_decomposition((0,1)))


        [Z(0),
         CNOT(wires=[0, 1]),
         SX(wires=[1]),
         RX(1.5707963267948966, wires=[0]),
         RY(1.5707963267948966, wires=[0]),
         RX(1.5707963267948966, wires=[0])]

        """
        pi = np.pi
        return [Z(wires=[wires[0]]), qml.CNOT(wires=[wires[0], wires[1]]), SX(wires=[wires[1]]), qml.RX(pi / 2, wires=[wires[0]]), qml.RY(pi / 2, wires=[wires[0]]), qml.RX(pi / 2, wires=[wires[0]])]

    def adjoint(self):
        return ECR(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)