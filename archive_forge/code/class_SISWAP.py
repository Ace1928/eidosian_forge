import cmath
from copy import copy
from functools import lru_cache
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane.operation import Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
class SISWAP(Operation):
    """SISWAP(wires)
    The square root of i-swap operator. Can also be accessed as ``qml.SQISW``

    .. math:: SISWAP = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1/ \\sqrt{2} & i/\\sqrt{2} & 0\\\\
            0 & i/ \\sqrt{2} & 1/ \\sqrt{2} & 0\\\\
            0 & 0 & 0 & 1
        \\end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    batch_size = None

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SISWAP.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.SISWAP.compute_matrix())
        [[1.+0.j          0.+0.j          0.+0.j  0.+0.j]
         [0.+0.j  0.70710678+0.j  0.+0.70710678j  0.+0.j]
         [0.+0.j  0.+0.70710678j  0.70710678+0.j  0.+0.j]
         [0.+0.j          0.+0.j          0.+0.j  1.+0.j]]
        """
        return np.array([[1, 0, 0, 0], [0, INV_SQRT2, INV_SQRT2 * 1j, 0], [0, INV_SQRT2 * 1j, INV_SQRT2, 0], [0, 0, 0, 1]])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.SISWAP.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.SISWAP.compute_eigvals())
        [0.70710678+0.70710678j 0.70710678-0.70710678j 1.+0.j 1.+0.j]
        """
        return np.array([INV_SQRT2 * (1 + 1j), INV_SQRT2 * (1 - 1j), 1, 1])

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.SISWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SISWAP.compute_decomposition((0,1)))
        [SX(wires=[0]),
        RZ(1.5707963267948966, wires=[0]),
        CNOT(wires=[0, 1]),
        SX(wires=[0]),
        RZ(5.497787143782138, wires=[0]),
        SX(wires=[0]),
        RZ(1.5707963267948966, wires=[0]),
        SX(wires=[1]),
        RZ(5.497787143782138, wires=[1]),
        CNOT(wires=[0, 1]),
        SX(wires=[0]),
        SX(wires=[1])]

        """
        return [SX(wires=wires[0]), qml.RZ(np.pi / 2, wires=wires[0]), qml.CNOT(wires=[wires[0], wires[1]]), SX(wires=wires[0]), qml.RZ(7 * np.pi / 4, wires=wires[0]), SX(wires=wires[0]), qml.RZ(np.pi / 2, wires=wires[0]), SX(wires=wires[1]), qml.RZ(7 * np.pi / 4, wires=wires[1]), qml.CNOT(wires=[wires[0], wires[1]]), SX(wires=wires[0]), SX(wires=wires[1])]

    def pow(self, z):
        z_mod4 = z % 4
        return [ISWAP(wires=self.wires)] if z_mod4 == 2 else super().pow(z_mod4)