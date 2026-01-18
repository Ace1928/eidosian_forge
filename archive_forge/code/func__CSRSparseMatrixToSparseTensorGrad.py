from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('CSRSparseMatrixToSparseTensor')
def _CSRSparseMatrixToSparseTensorGrad(op, *grads):
    """Gradient for csr_sparse_matrix_to_sparse_tensor op."""
    return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(indices=op.outputs[0], values=grads[1], dense_shape=op.outputs[2])