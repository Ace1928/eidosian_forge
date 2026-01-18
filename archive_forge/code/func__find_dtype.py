import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _find_dtype(value, preferred):
    """Returns the preferred dtype of value or preferred if preferred != None.

  This is used as an operator to pass over multiple objects in decreasing order
  of priority until there is a preferred dtype for one. For example, if you were
  adding three tensor-ish things (some tensors, some lists), and needed a
  preferred dtype, you could use this as:

  def adding(a, b, c, dtype = None):
    dtype = _find_dtype(a, dtype)
    dtype = _find_dtype(b, dtype)
    dtype = _find_dtype(c, dtype)
    if dtype is None:
      dtype = tf.float32
    ...Code continues here...

  Args:
    value: a list, value, RowPartition, or tensor.
    preferred: a given dtype. If not None, this will be returned.

  Returns:
    an optional dtype.
  """
    result = _find_dtype_helper(value, preferred)
    if result == dtypes.int64 or result == dtypes.int32 or result is None:
        return result
    raise ValueError('Illegal dtype: ' + str(result))