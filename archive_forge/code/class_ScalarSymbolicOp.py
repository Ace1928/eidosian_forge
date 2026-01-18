from abc import abstractmethod
from copy import copy
import numpy as np
import pennylane as qml
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.queuing import QueuingManager
class ScalarSymbolicOp(SymbolicOp):
    """Developer-facing base class for single-operator symbolic operators that contain a
    scalar coefficient.

    Args:
        base (~.operation.Operator): the base operation that is modified symbolicly
        scalar (float): the scalar coefficient
        id (str): custom label given to an operator instance, can be useful for some applications
            where the instance has to be identified

    This *developer-facing* class can serve as a parent to single base symbolic operators, such as
    :class:`~.ops.op_math.SProd` and :class:`~.ops.op_math.Pow`.
    """
    _name = 'ScalarSymbolicOp'

    def __init__(self, base, scalar: float, id=None):
        self.scalar = np.array(scalar) if isinstance(scalar, list) else scalar
        super().__init__(base, id=id)
        self._batch_size = _UNSET_BATCH_SIZE

    @property
    def batch_size(self):
        if self._batch_size is _UNSET_BATCH_SIZE:
            base_batch_size = self.base.batch_size
            if qml.math.ndim(self.scalar) == 0:
                self._batch_size = base_batch_size
            else:
                scalar_size = qml.math.size(self.scalar)
                if base_batch_size is not None and base_batch_size != scalar_size:
                    raise ValueError(f'Broadcasting was attempted but the broadcasted dimensions do not match: {scalar_size}, {base_batch_size}.')
                self._batch_size = scalar_size
        return self._batch_size

    @property
    def data(self):
        return (self.scalar, *self.base.data)

    @data.setter
    def data(self, new_data):
        self.scalar = new_data[0]
        self.base.data = new_data[1:]

    @property
    def has_matrix(self):
        return self.base.has_matrix or isinstance(self.base, qml.Hamiltonian)

    @property
    def hash(self):
        return hash((str(self.name), str(self.scalar), self.base.hash))

    @staticmethod
    @abstractmethod
    def _matrix(scalar, mat):
        """Scalar-matrix operation that doesn't take into account batching.

        ``ScalarSymbolicOp.matrix`` will call this method to compute the matrix for a single scalar
        and base matrix.

        Args:
            scalar (Union[int, float]): non-broadcasted scalar
            mat (ndarray): non-broadcasted matrix
        """

    def matrix(self, wire_order=None):
        """Representation of the operator as a matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        If the matrix depends on trainable parameters, the result
        will be cast in the same autodifferentiation framework as the parameters.

        A ``MatrixUndefinedError`` is raised if the base matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator.compute_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the
            operator's wires

        Returns:
            tensor_like: matrix representation
        """
        if isinstance(self.base, qml.Hamiltonian):
            base_matrix = qml.matrix(self.base)
        else:
            base_matrix = self.base.matrix()
        scalar_interface = qml.math.get_interface(self.scalar)
        scalar = self.scalar
        if scalar_interface == 'torch':
            base_matrix = qml.math.convert_like(base_matrix, self.scalar)
        elif scalar_interface == 'tensorflow':
            scalar = qml.math.cast(scalar, 'complex128')
            base_matrix = qml.math.cast(base_matrix, 'complex128')
        scalar_size = qml.math.size(scalar)
        if scalar_size != 1:
            if scalar_size == self.base.batch_size:
                mat = qml.math.stack([self._matrix(s, m) for s, m in zip(scalar, base_matrix)])
            else:
                mat = qml.math.stack([self._matrix(s, base_matrix) for s in scalar])
        elif self.base.batch_size is not None:
            mat = qml.math.stack([self._matrix(scalar, ar2) for ar2 in base_matrix])
        else:
            mat = self._matrix(scalar, base_matrix)
        return qml.math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)