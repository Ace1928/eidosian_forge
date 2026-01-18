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
def _select_class_id(ids, selected_id):
    """Filter all but `selected_id` out of `ids`.

  Args:
    ids: `int64` `Tensor` or `SparseTensor` of IDs.
    selected_id: Int id to select.

  Returns:
    `SparseTensor` of same dimensions as `ids`. This contains only the entries
    equal to `selected_id`.
  """
    ids = sparse_tensor.convert_to_tensor_or_sparse_tensor(ids)
    if isinstance(ids, sparse_tensor.SparseTensor):
        return sparse_ops.sparse_retain(ids, math_ops.equal(ids.values, selected_id))
    ids_shape = array_ops.shape(ids, out_type=dtypes.int64)
    ids_last_dim = array_ops.size(ids_shape) - 1
    filled_selected_id_shape = math_ops.reduced_shape(ids_shape, array_ops.reshape(ids_last_dim, [1]))
    filled_selected_id = array_ops.fill(filled_selected_id_shape, math_ops.cast(selected_id, dtypes.int64))
    result = sets.set_intersection(filled_selected_id, ids)
    return sparse_tensor.SparseTensor(indices=result.indices, values=result.values, dense_shape=ids_shape)