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
def _restore_sparse_tensors(stored_list, sparse_info_list):
    """Restore SparseTensors after dequeue in batch, batch_join, etc."""
    received_sequence = isinstance(stored_list, collections_abc.Sequence)
    if not received_sequence:
        stored_list = (stored_list,)
    tensors = [_restore_sparse(sparse_map_op=info.map_op, sparse_handles=array_ops.squeeze(s, [1]), rank=tensor_shape.dimension_value(info.rank + 1)) if info.sparse else s for s, info in zip(stored_list, sparse_info_list)]
    has_st = any((isinstance(x, sparse_tensor.SparseTensor) for x in tensors))
    if has_st:
        t_values = [x.values if isinstance(x, sparse_tensor.SparseTensor) else x for x in tensors]
        with_deps = lambda x: control_flow_ops.with_dependencies(t_values, x)
        ensure_restore_tensors = [sparse_tensor.SparseTensor(indices=with_deps(x.indices), values=with_deps(x.values), dense_shape=with_deps(x.dense_shape)) if isinstance(x, sparse_tensor.SparseTensor) else with_deps(x) for x in tensors]
    else:
        ensure_restore_tensors = tensors
    return ensure_restore_tensors if received_sequence else tensors[0]