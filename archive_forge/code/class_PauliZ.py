import cmath
from copy import copy
from functools import lru_cache
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane.operation import Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires
class PauliZ(Observable, Operation):
    """
    The Pauli Z operator

    .. math:: \\sigma_z = \\begin{bmatrix} 1 & 0 \\\\ 0 & -1\\end{bmatrix}.

    .. seealso:: The equivalent short-form alias :class:`~Z`

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    'int: Number of trainable parameters that the operator depends on.'
    basis = 'Z'
    batch_size = None
    _queue_category = '_ops'

    def __init__(self, *params, wires=None, id=None):
        super().__init__(*params, wires=wires, id=id)
        self._pauli_rep = qml.pauli.PauliSentence({qml.pauli.PauliWord({self.wires[0]: 'Z'}): 1.0})

    def __repr__(self):
        """String representation."""
        wire = self.wires[0]
        if isinstance(wire, str):
            return f"Z('{wire}')"
        return f'Z({wire})'

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or 'Z'

    @property
    def name(self):
        return 'PauliZ'

    @staticmethod
    @lru_cache()
    def compute_matrix():
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Z.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Z.compute_matrix())
        [[ 1  0]
         [ 0 -1]]
        """
        return np.array([[1, 0], [0, -1]])

    @staticmethod
    @lru_cache()
    def compute_sparse_matrix():
        return sparse.csr_matrix([[1, 0], [0, -1]])

    @staticmethod
    def compute_eigvals():
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Z.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.Z.compute_eigvals())
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

        .. seealso:: :meth:`~.Z.diagonalizing_gates`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.Z.compute_diagonalizing_gates(wires=[0]))
        []
        """
        return []

    @staticmethod
    def compute_decomposition(wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        .. seealso:: :meth:`~.Z.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.Z.compute_decomposition(0))
        [PhaseShift(3.141592653589793, wires=[0])]

        """
        return [qml.PhaseShift(np.pi, wires=wires)]

    def adjoint(self):
        return Z(wires=self.wires)

    def pow(self, z):
        z_mod2 = z % 2
        if z_mod2 == 0:
            return []
        if z_mod2 == 1:
            return [copy(self)]
        if abs(z_mod2 - 0.5) < 1e-06:
            return [S(wires=self.wires)]
        if abs(z_mod2 - 0.25) < 1e-06:
            return [T(wires=self.wires)]
        return [qml.PhaseShift(np.pi * z_mod2, wires=self.wires)]

    def _controlled(self, wire):
        return qml.CZ(wires=wire + self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi, 0.0, 0.0]