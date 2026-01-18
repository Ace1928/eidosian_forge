from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('DenseToCSRSparseMatrix')
def _DenseToCSRSparseMatrixGrad(op, grad):
    """Gradient for dense_to_csr_sparse_matrix op."""
    grad_values = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(grad, type=op.get_attr('T'))
    return (grad_values, None)