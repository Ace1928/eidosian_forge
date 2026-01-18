import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _replace_row_partitions(value, new_partitions):
    """Updates `value` to use `new_partitions` as its (outer) row partitions.

  This is used to ensure that all fields in a `StructuredTensor` use identical
  `RowPartition` objects for the shared dimensions.  In particular,
  `StructuredTensor.from_fields` first merges all of the row partitions from
  any fields, and then replaces the outer row partitions of all fields with
  the merged row partitions (using this function).

  Args:
    value: A `Tensor`, `RaggedTensor`, or `StructuredTensor`.
    new_partitions: A list of row-partitions that should be used by `value`.
      Must be equivalent to `value`'s current row partitions.

  Returns:
    A value that is equivalent to `value`, where outer row partitions have been
    replaced by `new_partitions`.
  """
    if isinstance(value, tensor.Tensor) or not new_partitions:
        return value
    elif isinstance(value, ragged_tensor.RaggedTensor):
        return ragged_tensor.RaggedTensor._from_row_partition(values=_replace_row_partitions(value.values, new_partitions[1:]), row_partition=new_partitions[0])
    else:
        assert isinstance(value, StructuredTensor)
        new_fields = dict(((k, _replace_row_partitions(v, new_partitions)) for k, v in value._fields.items()))
        return StructuredTensor._old_init(fields=new_fields, shape=value.shape, nrows=value.nrows(), row_partitions=tuple(new_partitions) + tuple(value.row_partitions[len(new_partitions):]))