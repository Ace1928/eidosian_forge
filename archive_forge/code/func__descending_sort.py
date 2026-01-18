import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _descending_sort(values, axis, return_argsort=False):
    """Sorts values in reverse using `top_k`.

  Args:
    values: Tensor of numeric values.
    axis: Index of the axis which values should be sorted along.
    return_argsort: If False, return the sorted values. If True, return the
      indices that would sort the values.

  Returns:
    The sorted values.
  """
    k = array_ops.shape(values)[axis]
    rank = array_ops.rank(values)
    static_rank = values.shape.ndims
    if axis == -1 or axis + 1 == values.get_shape().ndims:
        top_k_input = values
        transposition = None
    else:
        if axis < 0:
            axis += static_rank or rank
        if static_rank is not None:
            transposition = constant_op.constant(np.r_[np.arange(axis), [static_rank - 1], np.arange(axis + 1, static_rank - 1), [axis]], name='transposition')
        else:
            transposition = array_ops.tensor_scatter_update(math_ops.range(rank), [[axis], [rank - 1]], [rank - 1, axis])
        top_k_input = array_ops.transpose(values, transposition)
    values, indices = nn_ops.top_k(top_k_input, k)
    return_value = indices if return_argsort else values
    if transposition is not None:
        return_value = array_ops.transpose(return_value, transposition)
    return return_value