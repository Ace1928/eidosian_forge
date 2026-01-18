import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _dimension_sizes(x):
    """Gets the dimension sizes of a tensor `x`.

  If a size can be determined statically it is returned as an integer,
  otherwise as a tensor.

  If `x` is a scalar it is treated as rank 1 size 1.

  Args:
    x: A `Tensor`.

  Returns:
    Dimension sizes.
  """
    dynamic_shape = array_ops.shape(x)
    rank = x.get_shape().rank
    rank_is_known = rank is not None
    if rank_is_known and rank == 0:
        return (1,)
    if rank_is_known and rank > 0:
        static_shape = x.get_shape().as_list()
        sizes = [int(size) if size is not None else dynamic_shape[i] for i, size in enumerate(static_shape)]
        return sizes
    has_rank_zero = math_ops.equal(array_ops.rank(x), 0)
    return cond.cond(has_rank_zero, lambda: array_ops.constant([1]), lambda: dynamic_shape)