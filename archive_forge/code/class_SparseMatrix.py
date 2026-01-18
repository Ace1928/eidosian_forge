import abc
import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *
class SparseMatrix(metaclass=abc.ABCMeta):
    """Abstract class for sparse matrix types."""

    @abc.abstractmethod
    def __init__(self):
        self._eager_mode = context.executing_eagerly()

    @abc.abstractproperty
    def _matrix(self):
        pass

    @abc.abstractmethod
    def _from_matrix(self, matrix, handle_data=None):
        pass

    @abc.abstractmethod
    def to_dense(self):
        pass

    @abc.abstractmethod
    def to_sparse_tensor(self):
        pass

    @property
    def graph(self):
        return self._matrix.graph

    @property
    def shape(self):
        return dense_shape_and_type(self._matrix).shape

    @property
    def dtype(self):
        return dense_shape_and_type(self._matrix).dtype

    @property
    def eager_handle_data(self):
        """Return the matrix's handle data iff in eager mode."""
        return _get_handle_data(self._matrix) if self._eager_mode else None

    def conj(self):
        return self._from_matrix(math_ops.conj(self._matrix), self.eager_handle_data)

    def hermitian_transpose(self):
        """Return the hermitian transpose of the matrix."""
        return self._from_matrix(sm_ops.sparse_matrix_transpose(self._matrix, conjugate=True, type=self.dtype), self.eager_handle_data)

    def nnz(self):
        """Number of stored values, including explicit zeros."""
        return sm_ops.sparse_matrix_nnz(self._matrix)
    nonzero = nnz

    def sorted_indices(self):
        return self.to_sparse_tensor().indices

    def transpose(self):
        return self._from_matrix(sm_ops.sparse_matrix_transpose(self._matrix, type=self.dtype), self.eager_handle_data)