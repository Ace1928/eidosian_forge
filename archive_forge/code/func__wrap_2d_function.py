import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
def _wrap_2d_function(inputs, compute_op, dim=-1, name=None):
    """Helper function for ops that accept and return 2d inputs of same shape.

  It reshapes and transposes the inputs into a 2-D Tensor and then invokes
  the given function. The output would be transposed and reshaped back.
  If the given function returns a tuple of tensors, each of them will be
  transposed and reshaped.

  Args:
    inputs: A non-empty `Tensor`. Must be one of the following types: `half`,
      `float32`, `float64`.
    compute_op: The function to wrap. Must accept the input tensor as its first
      arugment, and a second keyword argument `name`.
    dim: The dimension softmax would be performed on. The default is -1 which
      indicates the last dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same shape as inputs. If compute_op returns multiple
      tensors, each of them have the same shape as the input.
  Raises:
    InvalidArgumentError: if `inputs` is empty or `dim` is beyond the last
      dimension of `inputs`.
  """

    def _swap_axis(input_tensor, dim_index, last_index, name=None):
        """Swaps logits's dim_index and last_index."""
        return array_ops.transpose(input_tensor, array_ops.concat([math_ops.range(dim_index), [last_index], math_ops.range(dim_index + 1, last_index), [dim_index]], 0), name=name)
    inputs = ops.convert_to_tensor(inputs)
    shape = inputs.get_shape()
    is_last_dim = dim == -1 or dim == shape.ndims - 1
    if is_last_dim:
        return compute_op(inputs, name=name)
    dim_val = dim
    if isinstance(dim, tensor_lib.Tensor):
        dim_val = tensor_util.constant_value(dim)
    if dim_val is not None and (not -shape.ndims <= dim_val < shape.ndims):
        raise errors_impl.InvalidArgumentError(None, None, f'`dim` must be in the range [{-shape.ndims}, {shape.ndims}) where {shape.ndims} is the number of dimensions in the input. Received: dim={dim_val}')
    ndims = array_ops.rank(inputs)
    if not isinstance(dim, tensor_lib.Tensor):
        if dim < 0:
            dim += ndims
    else:
        dim = array_ops.where(math_ops.less(dim, 0), dim + ndims, dim)
    input_rank = array_ops.rank(inputs)
    dim_axis = dim % shape.ndims
    inputs = _swap_axis(inputs, dim_axis, math_ops.subtract(input_rank, 1))

    def fix_output(output):
        output = _swap_axis(output, dim_axis, math_ops.subtract(input_rank, 1), name=name)
        output.set_shape(shape)
        return output
    outputs = compute_op(inputs)
    if isinstance(outputs, tuple):
        return tuple((fix_output(output) for output in outputs))
    else:
        return fix_output(outputs)