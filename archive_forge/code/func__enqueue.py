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
def _enqueue(queue, tensor_list, threads, enqueue_many, keep_input):
    """Enqueue `tensor_list` in `queue`."""
    if enqueue_many:
        enqueue_fn = queue.enqueue_many
    else:
        enqueue_fn = queue.enqueue
    if keep_input.shape.ndims == 1:
        enqueue_ops = [enqueue_fn(_select_which_to_enqueue(tensor_list, keep_input))] * threads
    else:
        enqueue_ops = [utils.smart_cond(keep_input, lambda: enqueue_fn(tensor_list), control_flow_ops.no_op)] * threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))