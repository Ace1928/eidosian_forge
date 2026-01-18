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
def _batch_join(tensors_list, batch_size, keep_input, capacity=32, enqueue_many=False, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, shared_name=None, name=None):
    """Helper function for `batch_join` and `maybe_batch_join`."""
    if context.executing_eagerly():
        raise ValueError('Input pipelines based on Queues are not supported when eager execution is enabled. Please use tf.data to ingest data into your model instead.')
    tensor_list_list = _as_tensor_list_list(tensors_list)
    with ops.name_scope(name, 'batch_join', _flatten(tensor_list_list) + [keep_input]) as name:
        tensor_list_list = _validate_join(tensor_list_list)
        keep_input = _validate_keep_input(keep_input, enqueue_many)
        tensor_list_list, sparse_info = _store_sparse_tensors_join(tensor_list_list, enqueue_many, keep_input)
        types = _dtypes(tensor_list_list)
        shapes = _shapes(tensor_list_list, shapes, enqueue_many)
        queue = _which_queue(dynamic_pad)(capacity=capacity, dtypes=types, shapes=shapes, shared_name=shared_name)
        _enqueue_join(queue, tensor_list_list, enqueue_many, keep_input)
        summary.scalar('fraction_of_%d_full' % capacity, math_ops.cast(queue.size(), dtypes.float32) * (1.0 / capacity))
        if allow_smaller_final_batch:
            dequeued = queue.dequeue_up_to(batch_size, name=name)
        else:
            dequeued = queue.dequeue_many(batch_size, name=name)
        dequeued = _restore_sparse_tensors(dequeued, sparse_info)
        return _as_original_type(tensors_list[0], dequeued)