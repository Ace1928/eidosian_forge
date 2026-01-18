import cmath
from copy import copy
from functools import lru_cache
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane.operation import Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
class PauliX(Observable, Operation):
    """
    The Pauli X operator

    .. math:: \\sigma_x = \\begin{bmatrix} 0 & 1 \\\\ 1 & 0\\end{bmatrix}.

    .. seealso:: The equivalent short-form alias :class:`~X`

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    'int: Number of wires that the operator acts on.'
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    basis = 'X'
    batch_size = None
    _queue_category = '_ops'

    def __init__(self, *params, wires=None, id=None):
        super().__init__(*params, wires=wires, id=id)
        self._pauli_rep = qml.pauli.PauliSentence({qml.pauli.PauliWord({self.wires[0]: 'X'}): 1.0})

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or 'X'

    def __repr__(self):
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"X('{wire}')"
        return f'X({wire})'

    @property
    def name(self):
        return 'PauliX'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.X.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.X.compute_matrix())
        [[0 1]
         [1 0]]
        """
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    @lru_cache()
    def compute_sparse_matrix():
        return sparse.csr_matrix([[0, 1], [1, 0]])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.X.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.X.compute_eigvals())
        [ 1 -1]
        """
        return pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires):
        """Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \\Sigma U^{\\dagger}` where
        :math:`\\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.X.diagonalizing_gates`.

        Args:
           wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
           list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.X.compute_diagonalizing_gates(wires=[0]))
        [Hadamard(wires=[0])]
        """
        return [Hadamard(wires=wires)]

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.


        .. seealso:: :meth:`~.X.decomposition`.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.X.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0]),
        RX(3.141592653589793, wires=[0]),
        PhaseShift(1.5707963267948966, wires=[0])]

        """
        return [qml.PhaseShift(np.pi / 2, wires=wires), qml.RX(np.pi, wires=wires), qml.PhaseShift(np.pi / 2, wires=wires)]

    def adjoint(self):
        return X(wires=self.wires)

    def pow(self, z):
        z_mod2 = z % 2
        if abs(z_mod2 - 0.5) < 1e-06:
            return [SX(wires=self.wires)]
        return super().pow(z_mod2)

    def _controlled(self, wire):
        return qml.CNOT(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi / 2, np.pi, -np.pi / 2]