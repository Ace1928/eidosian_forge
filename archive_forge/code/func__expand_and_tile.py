from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def _expand_and_tile(tensor, multiple, dim=0, name=None):
    """Slice `tensor` shape in 2, then tile along the sliced dimension.

  A new dimension is inserted in shape of `tensor` before `dim`, then values are
  tiled `multiple` times along the new dimension.

  Args:
    tensor: Input `Tensor` or `SparseTensor`.
    multiple: Integer, number of times to tile.
    dim: Integer, dimension along which to tile.
    name: Name of operation.

  Returns:
    `Tensor` result of expanding and tiling `tensor`.

  Raises:
    ValueError: if `multiple` is less than 1, or `dim` is not in
    `[-rank(tensor), rank(tensor)]`.
  """
    if multiple < 1:
        raise ValueError(f'Invalid argument multiple={multiple} for expand_and_tile  call. `multiple` must be an integer > 0')
    with ops.name_scope(name, 'expand_and_tile', (tensor, multiple, dim)) as scope:
        tensor = sparse_tensor.convert_to_tensor_or_sparse_tensor(tensor)
        if isinstance(tensor, sparse_tensor.SparseTensor):
            if dim < 0:
                expand_dims = array_ops.reshape(array_ops.size(tensor.dense_shape) + dim, [1])
            else:
                expand_dims = [dim]
            expanded_shape = array_ops.concat((array_ops.slice(tensor.dense_shape, [0], expand_dims), [1], array_ops.slice(tensor.dense_shape, expand_dims, [-1])), 0, name='expanded_shape')
            expanded = sparse_ops.sparse_reshape(tensor, shape=expanded_shape, name='expand')
            if multiple == 1:
                return expanded
            return sparse_ops.sparse_concat(dim - 1 if dim < 0 else dim, [expanded] * multiple, name=scope)
        expanded = array_ops.expand_dims(tensor, dim if dim >= 0 else dim - 1, name='expand')
        if multiple == 1:
            return expanded
        ones = array_ops.ones_like(array_ops.shape(tensor))
        tile_multiples = array_ops.concat((ones[:dim], (multiple,), ones[dim:]), 0, name='multiples')
        return array_ops.tile(expanded, tile_multiples, name=scope)