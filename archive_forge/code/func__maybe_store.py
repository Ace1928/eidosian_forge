from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _maybe_store(t, shared_map_op):
    """Store Sparse tensor, if necessary."""
    if not isinstance(t, sparse_tensor.SparseTensor):
        return t
    map_op_name = shared_map_op.name if shared_map_op else None

    def _maybe_store_sparse(t, map_op_name, keep_input):
        """Conditionally store a single sparse Tensor."""
        return utils.smart_cond(keep_input, lambda: _store_sparse(t, shared_name=map_op_name), lambda: constant_op.constant(-1, dtypes.int64))

    def _maybe_store_many_sparse(t, map_op_name, keep_input):
        """Conditionally store multiple sparse Tensors."""
        out_tensor = utils.smart_cond(keep_input, lambda: _store_many_sparse(t, shared_name=map_op_name), lambda: -1 * array_ops.ones(array_ops.shape(t)[0:1], dtypes.int64))
        out_tensor.set_shape([None])
        return out_tensor

    def _sparse_values_to_keep(t, keep_input):
        """Convert a per-row `keep_input` vector to a per-value one."""
        row_values = t.indices[:, 0]
        return array_ops.gather(keep_input, row_values)
    if keep_input.shape.ndims == 1:
        t = sparse_ops.sparse_retain(t, _sparse_values_to_keep(t, keep_input))
        store_f = lambda t, name, _: _store_many_sparse(t, shared_name=name)
    elif enqueue_many:
        store_f = _maybe_store_many_sparse
    else:
        store_f = _maybe_store_sparse
    return store_f(t, map_op_name, keep_input)