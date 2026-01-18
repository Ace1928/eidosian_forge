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
def _validate_keep_input(keep_input, enqueue_many):
    """Validate `keep_input` argument to conditional batching functions."""
    keep_input = ops.convert_to_tensor(keep_input)
    if keep_input.shape.ndims is None:
        raise ValueError('`keep_input` dimensions must be known at graph construction.')
    if not enqueue_many and keep_input.shape.ndims == 1:
        raise ValueError('`keep_input` cannot be a vector when `enqueue_many=False`.')
    if keep_input.shape.ndims > 1:
        raise ValueError('`keep_input` must be 0 or 1 dimensions.')
    return keep_input