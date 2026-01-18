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
def _assert_concat_compatible_structured_tensors(values):
    """Sometimes raises an error if concat doesn't make sense statically on values.

  values must be a sequence, and each element in values must be a structured
  tensor, and must have the same paths. Additionally, each path that is a
  submessage must have the same rank.

  These constraints are sufficient for concat on the fields to be the same
  as concat on structured tensors. This is meant to capture scenarios like
  paths that are not in the first structured tensor, but are in later
  structured tensors, which will just be ignored by the recursive algorithm.

  If the rank of a submessage was different for two structured tensors,
  then that is also a non-sensical merge.

  Note that all of these checks are static, as paths and submessage ranks
  are known.

  Args:
    values: a Sequence of StructuredTensors.

  Raises:
    ValueError: if there is any inconsistency as described above.
  """
    if not isinstance(values, Sequence):
        raise ValueError('values must be a list of StructuredTensors (not a list)')
    if not values:
        raise ValueError('values must not be an empty list')
    for st in values:
        if not isinstance(st, StructuredTensor):
            raise ValueError('values must be a list of StructuredTensors')
    _assert_all_paths_match(values)
    _assert_all_ranks_match(values)