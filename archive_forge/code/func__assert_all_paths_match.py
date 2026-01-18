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
def _assert_all_paths_match(values):
    """Raises an error if the paths are not identical."""
    paths = [_get_all_paths(st) for st in values]
    path_diff = set()
    for other_paths in paths[1:]:
        path_diff = path_diff.union(paths[0].symmetric_difference(other_paths))
    if path_diff:
        raise ValueError('Some paths are present in some, but not all, structured tensors: %r' % (path_diff,))