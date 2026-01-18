from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
def _PruneSparseTensor(unpruned, pruned_pattern):
    """Helper function to prune COO sparse tensor.

  Given two sparse tensors 'unpruned' and 'pruned_pattern', generates another
  sparse tensor with indices and values fron 'unpruned' only if its indices also
  occur in pruned_pattern.

  Args:
    unpruned: COO matrix with unpruned indices
    pruned_pattern: COO matrix with pruned pattern.

  TODO(tabakg): This is far from optimal. Consider a C++ implementation.

  Returns:
    Indices, values, and dense_shape of the pruned matrix.
  """
    pruned_indices = sparse_ops.sparse_reshape(pruned_pattern, shape=(-1,)).indices[..., 0]
    unpruned_indices = sparse_ops.sparse_reshape(unpruned, shape=(-1,)).indices[..., 0]
    best_match = array_ops.searchsorted(unpruned_indices, pruned_indices)
    keep_indices = array_ops.gather(best_match, array_ops.where(math_ops.equal(array_ops.gather(unpruned_indices, best_match), pruned_indices)))
    return (array_ops.gather_nd(unpruned.indices, keep_indices), array_ops.gather_nd(unpruned.values, keep_indices), pruned_pattern.dense_shape)