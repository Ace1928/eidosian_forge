from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
@ops.RegisterGradient('SparseMatrixMul')
def _SparseMatrixMulGrad(op, grad):
    """Gradient for sparse_matrix_mul op."""
    del op
    del grad
    raise NotImplementedError