import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane.operation import Observable
from pennylane.ops.qubit import Hermitian
from pennylane.ops.qutrit import QutritUnitary
class THermitian(Hermitian):
    """An arbitrary Hermitian observable for qutrits.

    For a Hermitian matrix :math:`A`, the expectation command returns the value

    .. math::
        \\braket{A} = \\braketT{\\psi}{\\cdots \\otimes I\\otimes A\\otimes I\\cdots}{\\psi}

    where :math:`A` acts on the requested wires.

    If acting on :math:`N` wires, then the matrix :math:`A` must be of size
    :math:`3^N\\times 3^N`.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        A (array): square Hermitian matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str or None): String representing the operation (optional)

    .. note::
        :class:`Hermitian` cannot be used with qutrit devices due to its use of
        :class:`QubitUnitary` in :meth:`~.Hermitian.compute_diagonalizing_gates`.

    """
    _num_basis_states = 3
    _eigs = {}

    @staticmethod
    def compute_matrix(A):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.THermitian.matrix`

        Args:
            A (tensor_like): Hermitian matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[6+0j, 1-2j, 0],[1+2j, -1, 0], [0, 0, 1]])
        >>> qml.THermitian.compute_matrix(A)
        [[ 6.+0.j  1.-2.j  0.+0.j]
         [ 1.+2.j -1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j]]
        """
        return Hermitian.compute_matrix(A)

    @property
    def eigendecomposition(self):
        """Return the eigendecomposition of the matrix specified by the Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the eigenvectors of the
                Hermitian observable
        """
        Hmat = self.matrix()
        Hmat = qml.math.to_numpy(Hmat)
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in THermitian._eigs:
            w, U = np.linalg.eigh(Hmat)
            THermitian._eigs[Hkey] = {'eigvec': U, 'eigval': w}
        return THermitian._eigs[Hkey]

    @staticmethod
    def compute_diagonalizing_gates(eigenvectors, wires):
        """Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \\Sigma U^{\\dagger}` where
        :math:`\\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.THermitian.diagonalizing_gates`.

        Args:
            eigenvectors (array): eigenvectors of the operator, as extracted from op.eigendecomposition["eigvec"]
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> A = np.array([[-6, 2 + 1j, 0], [2 - 1j, 0, 0], [0, 0, 1]])
        >>> _, evecs = np.linalg.eigh(A)
        >>> qml.THermitian.compute_diagonalizing_gates(evecs, wires=[0])
        [QutritUnitary(tensor([[-0.94915323-0.j    0.1407893 +0.2815786j  -0.        -0.j  ]
                               [ 0.31481445-0.j    0.42447423+0.84894846j  0.        -0.j  ]
                               [ 0.        -0.j    0.        -0.j          1.        -0.j  ]], requires_grad=True), wires=[0])]

        """
        return [QutritUnitary(eigenvectors.conj().T, wires=wires)]