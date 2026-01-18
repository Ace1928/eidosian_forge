import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _ascending_sort(values, axis, return_argsort=False):
    """Sorts values in ascending order.

  Args:
    values: Tensor of numeric values.
    axis: Index of the axis which values should be sorted along.
    return_argsort: If False, return the sorted values. If True, return the
      indices that would sort the values.

  Returns:
    The sorted values.
  """
    dtype = values.dtype
    if dtype.is_unsigned:
        offset = dtype.max
        values_or_indices = _descending_sort(offset - values, axis, return_argsort)
        return values_or_indices if return_argsort else offset - values_or_indices
    elif dtype.is_integer:
        values_or_indices = _descending_sort(-values - 1, axis, return_argsort)
        return values_or_indices if return_argsort else -values_or_indices - 1
    else:
        values_or_indices = _descending_sort(-values, axis, return_argsort)
        return values_or_indices if return_argsort else -values_or_indices