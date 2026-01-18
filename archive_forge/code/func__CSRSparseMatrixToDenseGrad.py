from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('CSRSparseMatrixToDense')
def _CSRSparseMatrixToDenseGrad(op, grad):
    """Gradient for csr_sparse_matrix_to_dense op."""
    coo_sparse_tensor = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(op.inputs[0], type=grad.dtype)
    return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(indices=coo_sparse_tensor.indices, values=array_ops.gather_nd(grad, coo_sparse_tensor.indices), dense_shape=grad.shape)