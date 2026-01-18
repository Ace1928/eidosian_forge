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
def _pretty_print(data_item, summarize):
    """Format a data item for use in an error message in eager mode.

  Args:
    data_item: One of the items in the "data" argument to an assert_* function.
      Can be a Tensor or a scalar value.
    summarize: How many elements to retain of each tensor-valued entry in data.

  Returns:
    An appropriate string representation of data_item
  """
    if isinstance(data_item, tensor_lib.Tensor):
        arr = data_item.numpy()
        if np.isscalar(arr):
            return str(arr)
        else:
            flat = arr.reshape((-1,))
            lst = [str(x) for x in flat[:summarize]]
            if len(lst) < flat.size:
                lst.append('...')
            return str(lst)
    else:
        return str(data_item)