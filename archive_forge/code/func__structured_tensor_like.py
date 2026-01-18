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
def _structured_tensor_like(t):
    """Create a StructuredTensor with the shape of a (composite) tensor."""
    if isinstance(t, tensor_lib.Tensor):
        return _structured_tensor_from_dense_tensor(t)
    if ragged_tensor.is_ragged(t):
        return StructuredTensor.from_fields({}, shape=t.get_shape(), row_partitions=_all_nested_row_partitions(t))
    return StructuredTensor.from_fields({}, shape=t.shape, row_partitions=t.row_partitions, nrows=t.nrows())