from copy import copy
from collections.abc import Sequence
import numpy as np
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.wires import Wires
from .matrix_ops import QubitUnitary
class StateVectorProjector(Projector):
    """Observable corresponding to the state projector :math:`P=\\ket{\\phi}\\bra{\\phi}`, where
    :math:`\\phi` denotes a state."""

    def __init__(self, state, wires, id=None):
        wires = Wires(wires)
        state = list(qml.math.toarray(state))
        super().__init__(state, wires=wires, id=id)

    def __new__(cls, *_, **__):
        return object.__new__(cls)

    def label(self, decimals=None, base_label=None, cache=None):
        """A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label.
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing.

        Returns:
            str: label to use in drawings.

        **Example:**

        >>> state_vector = np.array([0, 1, 1, 0])/np.sqrt(2)
        >>> qml.Projector(state_vector, wires=(0, 1)).label()
        'P'
        >>> qml.Projector(state_vector, wires=(0, 1)).label(base_label="hi!")
        'hi!'
        >>> dev = qml.device("default.qubit", wires=1)
        >>> @qml.qnode(dev)
        >>> def circuit(state):
        ...     return qml.expval(qml.Projector(state, [0]))
        >>> print(qml.draw(circuit)([1, 0]))
        0: ───┤  <|0⟩⟨0|>
        >>> print(qml.draw(circuit)(np.array([1, 1]) / np.sqrt(2)))
        0: ───┤  <P(M0)>
        M0 =
        [0.70710678 0.70710678]

        """
        if base_label is not None:
            return base_label
        state_vector = self.parameters[0]
        n_wires = int(qml.math.log2(len(state_vector)))
        basis_state_idx = qml.math.nonzero(state_vector)[0]
        if len(basis_state_idx) == 1:
            basis_string = f'{basis_state_idx[0]:0{n_wires}b}'
            return f'|{basis_string}⟩⟨{basis_string}|'
        if cache is None or not isinstance(cache.get('matrices', None), list):
            return 'P'
        mat_num = len(cache['matrices'])
        cache['matrices'].append(self.parameters[0])
        return f'P(M{mat_num})'

    @staticmethod
    def compute_matrix(state_vector):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Projector.matrix`

        Args:
            state_vector (Iterable): state vector to project on

        Returns:
            ndarray: matrix

        **Example**

        The projector of the state :math:`\\frac{1}{\\sqrt{2}}(\\ket{01}+\\ket{10})`

        >>> StateVectorProjector.compute_matrix([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
        [[0. 0.  0.  0.]
         [0. 0.5 0.5 0.]
         [0. 0.5 0.5 0.]
         [0. 0.  0.  0.]]
        """
        return qml.math.outer(state_vector, qml.math.conj(state_vector))

    @staticmethod
    def compute_eigvals(state_vector):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.StateVectorProjector.eigvals`

        Args:
            state_vector (Iterable): state vector to project on

        Returns:
            array: eigenvalues

        **Example**

        >>> StateVectorProjector.compute_eigvals([0, 0, 1, 0])
        array([1, 0, 0, 0])
        """
        w = qml.math.zeros_like(state_vector)
        w[0] = 1
        return w

    @staticmethod
    def compute_diagonalizing_gates(state_vector, wires):
        """Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \\Sigma U^{\\dagger}` where
        :math:`\\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.StateVectorProjector.diagonalizing_gates`.

        Args:
            state_vector (Iterable): state vector that the operator projects on.
            wires (Iterable[Any], Wires): wires that the operator acts on.
        Returns:
            list[.Operator]: list of diagonalizing gates.

        **Example**

        >>> state_vector = np.array([1., 1j])/np.sqrt(2)
        >>> StateVectorProjector.compute_diagonalizing_gates(state_vector, wires=[0])
        [QubitUnitary(array([[ 0.70710678+0.j        ,  0.        -0.70710678j],
                             [ 0.        +0.70710678j, -0.70710678+0.j        ]]), wires=[0])]
        """
        phase = qml.math.exp(-1j * qml.math.angle(state_vector[0]))
        psi = phase * state_vector
        denominator = qml.math.sqrt(2 + 2 * psi[0])
        psi = qml.math.set_index(psi, 0, psi[0] + 1)
        psi /= denominator
        u = 2 * qml.math.outer(psi, qml.math.conj(psi)) - qml.math.eye(len(psi))
        return [QubitUnitary(u, wires=wires)]