from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
def _PrunedDenseMatrixMultiplication(a, b, indices, transpose_a=False, adjoint_a=False, transpose_b=False, adjoint_b=False):
    """Multiplies two dense matrices at selected indices.

  The two inputs `a` and `b` must have matching rank (2 or 3). If using rank 3,
  the first rank is used for the batch number. The last two dimensions should
  also be compatible for matrix multiplication.

  TODO(tabakg): Consider C++ implementation. There is also a more efficient way
  to handle transposes here.

  Args:
    a: The left dense matrix (or batched matrices).
    b: The right dense matrix (or batched matrices).
    indices: The selected output indices where values should be produced. Other
      indices will be pruned (not computed in the first place). Indices are
      specified as a tensor of shape (length, rank), where length is the number
      of entries and rank is the rank of the dense inputs (2 or 3).
    transpose_a: Whether to transpose a.
    adjoint_a: Whether to take the conjugate transpose of a.
    transpose_b: Whether to transpose b.
    adjoint_b: Whether to take the conjugate transpose of b.

  Returns:
    A CSR matrix.
  """
    transpose_a = transpose_a or adjoint_a
    transpose_b = transpose_b or adjoint_b
    a = math_ops.conj(a) if adjoint_a else a
    b = math_ops.conj(b) if adjoint_b else b
    rank = len(a.shape)
    dense_shape = (a.shape[-1] if transpose_a else a.shape[-2], b.shape[-2] if transpose_b else b.shape[-1])
    if rank == 2:
        rows = indices[:, 0]
        cols = indices[:, 1]
        transpose = array_ops.transpose
        gather_op = array_ops.gather
    elif rank == 3:
        dense_shape = (a.shape[0],) + dense_shape
        rows = indices[:, :2]
        cols = array_ops_stack.stack([indices[:, 0], indices[:, 2]], axis=1)
        transpose = lambda x: array_ops.transpose(x, perm=[0, 2, 1])
        gather_op = array_ops.gather_nd
    a_rows = gather_op(transpose(a) if transpose_a else a, indices=rows)
    b_cols = gather_op(b if transpose_b else transpose(b), indices=cols)
    values = math_ops.reduce_sum(a_rows * b_cols, axis=1)
    return sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(indices=indices, values=values, dense_shape=dense_shape)