from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('SparseTensorToCSRSparseMatrix')
def _SparseTensorToCSRSparseMatrixGrad(op, grad):
    """Gradient for sparse_tensor_to_csr_sparse_matrix op."""
    grad_values = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(grad, type=op.get_attr('T')).values
    return (None, grad_values, None)