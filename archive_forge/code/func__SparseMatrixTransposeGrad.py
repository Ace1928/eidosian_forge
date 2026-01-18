from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('SparseMatrixTranspose')
def _SparseMatrixTransposeGrad(op, grad):
    """Gradient for sparse_matrix_transpose op."""
    return sparse_csr_matrix_ops.sparse_matrix_transpose(grad, type=op.get_attr('type'), conjugate=op.get_attr('conjugate'))