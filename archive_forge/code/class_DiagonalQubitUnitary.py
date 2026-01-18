import warnings
from itertools import product
import numpy as np
from scipy.linalg import fractional_matrix_power
from pennylane.math import norm, cast, eye, zeros, transpose, conj, sqrt, sqrt_matrix
from pennylane import numpy as pnp
import pennylane as qml
from pennylane.operation import AnyWires, DecompositionUndefinedError, Operation
from pennylane.wires import Wires
class DiagonalQubitUnitary(Operation):
    """DiagonalQubitUnitary(D, wires)
    Apply an arbitrary diagonal unitary matrix with a dimension that is a power of two.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe: None

    Args:
        D (array[complex]): diagonal of unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    ndim_params = (1,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    @staticmethod
    def compute_matrix(D):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.matrix`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_matrix(torch.tensor([1, -1]))
        tensor([[ 1,  0],
                [ 0, -1]])
        """
        D = qml.math.asarray(D)
        if not qml.math.is_abstract(D) and (not qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D))):
            raise ValueError('Operator must be unitary.')
        if qml.math.ndim(D) == 2:
            return qml.math.stack([qml.math.diag(_D) for _D in D])
        return qml.math.diag(D)

    @staticmethod
    def compute_eigvals(D):
        """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.eigvals`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_eigvals(torch.tensor([1, -1]))
        tensor([ 1, -1])
        """
        D = qml.math.asarray(D)
        if not (qml.math.is_abstract(D) or qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D))):
            raise ValueError('Operator must be unitary.')
        return D

    @staticmethod
    def compute_decomposition(D, wires):
        """Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \\dots O_n.

        ``DiagonalQubitUnitary`` decomposes into :class:`~.QubitUnitary`, :class:`~.RZ`,
        :class:`~.IsingZZ`, and/or :class:`~.MultiRZ` depending on the number of wires.

        .. note::

            The parameters of the decomposed operations are cast to the ``complex128`` dtype
            as real dtypes can lead to ``NaN`` values in the decomposition.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.decomposition`.

        Args:
            D (tensor_like): diagonal of the matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> diag = np.exp(1j * np.array([0.4, 2.1, 0.5, 1.8]))
        >>> qml.DiagonalQubitUnitary.compute_decomposition(diag, wires=[0, 1])
        [QubitUnitary(array([[0.36235775+0.93203909j, 0.        +0.j        ],
         [0.        +0.j        , 0.36235775+0.93203909j]]), wires=[0]),
         RZ(1.5000000000000002, wires=[1]),
         RZ(-0.10000000000000003, wires=[0]),
         IsingZZ(0.2, wires=[0, 1])]

        """
        n = len(wires)
        D_casted = qml.math.cast(D, 'complex128')
        phases = qml.math.real(qml.math.log(D_casted) * -1j)
        coeffs = _walsh_hadamard_transform(phases, n).T
        global_phase = qml.math.exp(1j * coeffs[0])
        coeffs = coeffs * -2.0
        ops = [QubitUnitary(qml.math.tensordot(global_phase, qml.math.eye(2), axes=0), wires[0])]
        for wire0 in range(n):
            ops.append(qml.RZ(coeffs[1 << wire0], n - 1 - wire0))
            ops.extend((qml.IsingZZ(coeffs[(1 << wire0) + (1 << wire1)], [n - 1 - wire0, n - 1 - wire1]) for wire1 in range(wire0)))
        ops.extend((qml.MultiRZ(c, [wires[k] for k in np.where(term)[0]]) for c, term in zip(coeffs, product((0, 1), repeat=n)) if sum(term) > 2))
        return ops

    def adjoint(self):
        return DiagonalQubitUnitary(qml.math.conj(self.parameters[0]), wires=self.wires)

    def pow(self, z):
        cast_data = qml.math.cast(self.data[0], np.complex128)
        return [DiagonalQubitUnitary(cast_data ** z, wires=self.wires)]

    def _controlled(self, control):
        return DiagonalQubitUnitary(qml.math.hstack([np.ones_like(self.parameters[0]), self.parameters[0]]), wires=control + self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'U', cache=cache)