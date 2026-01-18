import warnings
from itertools import product
import numpy as np
from scipy.linalg import fractional_matrix_power
from pennylane.math import norm, cast, eye, zeros, transpose, conj, sqrt, sqrt_matrix
from pennylane import numpy as pnp
import pennylane as qml
from pennylane.operation import AnyWires, DecompositionUndefinedError, Operation
from pennylane.wires import Wires
class BlockEncode(Operation):
    """BlockEncode(A, wires)
    Construct a unitary :math:`U(A)` such that an arbitrary matrix :math:`A`
    is encoded in the top-left block.

    .. math::

        \\begin{align}
             U(A) &=
             \\begin{bmatrix}
                A & \\sqrt{I-AA^\\dagger} \\\\
                \\sqrt{I-A^\\dagger A} & -A^\\dagger
            \\end{bmatrix}.
        \\end{align}

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        A (tensor_like): a general :math:`(n \\times m)` matrix to be encoded
        wires (Iterable[int, str], Wires): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    We can define a matrix and a block-encoding circuit as follows:

    >>> A = [[0.1,0.2],[0.3,0.4]]
    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.BlockEncode(A, wires=range(2))
    ...     return qml.state()

    We can see that :math:`A` has been block encoded in the matrix of the circuit:

    >>> print(qml.matrix(example_circuit)())
    [[ 0.1         0.2         0.97283788 -0.05988708]
     [ 0.3         0.4        -0.05988708  0.86395228]
     [ 0.94561648 -0.07621992 -0.1        -0.3       ]
     [-0.07621992  0.89117368 -0.2        -0.4       ]]

    We can also block-encode a non-square matrix and check the resulting unitary matrix:

    >>> A = [[0.2, 0, 0.2],[-0.2, 0.2, 0]]
    >>> op = qml.BlockEncode(A, wires=range(3))
    >>> print(np.round(qml.matrix(op), 2))
    [[ 0.2   0.    0.2   0.96  0.02  0.    0.    0.  ]
     [-0.2   0.2   0.    0.02  0.96  0.    0.    0.  ]
     [ 0.96  0.02 -0.02 -0.2   0.2   0.    0.    0.  ]
     [ 0.02  0.98  0.   -0.   -0.2   0.    0.    0.  ]
     [-0.02  0.    0.98 -0.2  -0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.    1.    0.    0.  ]
     [ 0.    0.    0.    0.    0.    0.    1.    0.  ]
     [ 0.    0.    0.    0.    0.    0.    0.    1.  ]]

    .. note::
        If the operator norm of :math:`A`  is greater than 1, we normalize it to ensure
        :math:`U(A)` is unitary. The normalization constant can be
        accessed through :code:`op.hyperparameters["norm"]`.

        Specifically, the norm is computed as the maximum of
        :math:`\\| AA^\\dagger \\|` and
        :math:`\\| A^\\dagger A \\|`.
    """
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    num_wires = AnyWires
    'int: Number of wires that the operator acts on.'
    ndim_params = (2,)
    'tuple[int]: Number of dimensions per trainable parameter that the operator depends on.'
    grad_method = None
    'Gradient computation method.'

    def __init__(self, A, wires, id=None):
        wires = Wires(wires)
        shape_a = qml.math.shape(A)
        if shape_a == () or all((x == 1 for x in shape_a)):
            A = qml.math.reshape(A, [1, 1])
            normalization = qml.math.abs(A)
            subspace = (1, 1, 2 ** len(wires))
        else:
            if len(shape_a) == 1:
                A = qml.math.reshape(A, [1, len(A)])
                shape_a = qml.math.shape(A)
            normalization = qml.math.maximum(norm(A @ qml.math.transpose(qml.math.conj(A)), ord=pnp.inf), norm(qml.math.transpose(qml.math.conj(A)) @ A, ord=pnp.inf))
            subspace = (*shape_a, 2 ** len(wires))
        A = qml.math.array(A) / qml.math.maximum(normalization, qml.math.ones_like(normalization))
        if subspace[2] < subspace[0] + subspace[1]:
            raise ValueError(f'Block encoding a ({subspace[0]} x {subspace[1]}) matrix requires a Hilbert space of size at least ({subspace[0] + subspace[1]} x {subspace[0] + subspace[1]}). Cannot be embedded in a {len(wires)} qubit system.')
        super().__init__(A, wires=wires, id=id)
        self.hyperparameters['norm'] = normalization
        self.hyperparameters['subspace'] = subspace

    def _flatten(self):
        return (self.data, (self.wires, ()))

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.BlockEncode.matrix`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute


        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[0.1,0.2],[0.3,0.4]])
        >>> A
        tensor([[0.1, 0.2],
                [0.3, 0.4]])
        >>> qml.BlockEncode.compute_matrix(A, subspace=[2,2,4])
        array([[ 0.1       ,  0.2       ,  0.97283788, -0.05988708],
               [ 0.3       ,  0.4       , -0.05988708,  0.86395228],
               [ 0.94561648, -0.07621992, -0.1       , -0.3       ],
               [-0.07621992,  0.89117368, -0.2       , -0.4       ]])
        """
        A = params[0]
        n, m, k = hyperparams['subspace']
        shape_a = qml.math.shape(A)

        def _stack(lst, h=False, like=None):
            if like == 'tensorflow':
                axis = 1 if h else 0
                return qml.math.concat(lst, like=like, axis=axis)
            return qml.math.hstack(lst) if h else qml.math.vstack(lst)
        interface = qml.math.get_interface(A)
        if qml.math.sum(shape_a) <= 2:
            col1 = _stack([A, sqrt(1 - A * conj(A))], like=interface)
            col2 = _stack([sqrt(1 - A * conj(A)), -conj(A)], like=interface)
            u = _stack([col1, col2], h=True, like=interface)
        else:
            d1, d2 = shape_a
            col1 = _stack([A, sqrt_matrix(cast(eye(d2, like=A), A.dtype) - qml.math.transpose(conj(A)) @ A)], like=interface)
            col2 = _stack([sqrt_matrix(cast(eye(d1, like=A), A.dtype) - A @ transpose(conj(A))), -transpose(conj(A))], like=interface)
            u = _stack([col1, col2], h=True, like=interface)
        if n + m < k:
            r = k - (n + m)
            col1 = _stack([u, zeros((r, n + m), like=A)], like=interface)
            col2 = _stack([zeros((n + m, r), like=A), eye(r, like=A)], like=interface)
            u = _stack([col1, col2], h=True, like=interface)
        return u

    def adjoint(self):
        A = self.parameters[0]
        return BlockEncode(qml.math.transpose(qml.math.conj(A)), wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or 'BlockEncode', cache=cache)