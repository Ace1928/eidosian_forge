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
def _assert_same_base_type(items, expected_type=None):
    """Asserts all items are of the same base type.

  Args:
    items: List of graph items (e.g., `Variable`, `Tensor`, `SparseTensor`,
        `Operation`, or `IndexedSlices`). Can include `None` elements, which
        will be ignored.
    expected_type: Expected type. If not specified, assert all items are
        of the same base type.

  Returns:
    Validated type, or none if neither expected_type nor items provided.

  Raises:
    ValueError: If any types do not match.
  """
    original_expected_type = expected_type
    mismatch = False
    for item in items:
        if item is not None:
            item_type = item.dtype.base_dtype
            if not expected_type:
                expected_type = item_type
            elif expected_type != item_type:
                mismatch = True
                break
    if mismatch:
        expected_type = original_expected_type
        original_item_str = None
        for item in items:
            if item is not None:
                item_type = item.dtype.base_dtype
                if not expected_type:
                    expected_type = item_type
                    original_item_str = item.name if hasattr(item, 'name') else str(item)
                elif expected_type != item_type:
                    raise ValueError('%s, type=%s, must be of the same type (%s)%s.' % (item.name if hasattr(item, 'name') else str(item), item_type, expected_type, ' as %s' % original_item_str if original_item_str else ''))
        return expected_type
    else:
        return expected_type