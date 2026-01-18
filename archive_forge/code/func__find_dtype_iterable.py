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
def _find_dtype_iterable(iterable: Iterable[Any], dtype: Optional[dtypes.DType]) -> Optional[dtypes.DType]:
    """Find the preferred dtype of a list of objects.

  This will go over the iterable, and use the first object with a preferred
  dtype. The dtype passed has highest priority if it is not None.

  Args:
    iterable: an iterable with things that might have a dtype.
    dtype: an overriding dtype, or None.

  Returns:
    an optional dtype.
  """
    if dtype is not None:
        return dtype
    for x in iterable:
        dtype = _find_dtype(x, dtype)
    return dtype