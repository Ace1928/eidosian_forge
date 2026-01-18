import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _sort_or_argsort(values, axis, direction, return_argsort):
    """Internal sort/argsort implementation.

  Args:
    values: The input values.
    axis: The axis along which to sort.
    direction: 'ASCENDING' or 'DESCENDING'.
    return_argsort: Whether to return the argsort result.

  Returns:
    Either the sorted values, or the indices of the sorted values in the
        original tensor. See the `sort` and `argsort` docstrings.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  """
    if direction not in _SORT_IMPL:
        valid_directions = ', '.join(sorted(_SORT_IMPL.keys()))
        raise ValueError(f'Argument `direction` should be one of {valid_directions}. Received: direction={direction}')
    axis = framework_ops.convert_to_tensor(axis, name='axis')
    axis_static = tensor_util.constant_value(axis)
    if axis.shape.ndims not in (None, 0) or axis_static is None:
        raise ValueError(f'Argument `axis` must be a constant scalar. Received: axis={axis}.')
    axis_static = int(axis_static)
    values = framework_ops.convert_to_tensor(values, name='values')
    return _SORT_IMPL[direction](values, axis_static, return_argsort)