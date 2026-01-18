from typing import Sequence
from tensorflow.core.config import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
def _get_all_paths(st):
    """Get all the paths from a StructuredTensor."""
    fields = st.field_names()
    all_paths = {()}
    for k in fields:
        v = st.field_value(k)
        if isinstance(v, StructuredTensor):
            all_paths = all_paths.union([(k,) + p for p in _get_all_paths(v)])
        else:
            all_paths.add((k,))
    return all_paths