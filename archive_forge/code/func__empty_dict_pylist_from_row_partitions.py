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
def _empty_dict_pylist_from_row_partitions(row_partitions, nrows):
    """Returns a python list of empty dicts from the given row partitions.

  Args:
    row_partitions: The row-partitions describing the ragged shape of the
      result.
    nrows: The number of rows in the outermost row-partition.  (Or if
      `len(row_partitions)==0`, then the number of empty dicts to return.)

  Returns:
    A nested python list whose leaves (if any) are empty python dicts.
  """
    if not row_partitions:
        return [{} for _ in range(nrows)]
    else:
        values = _empty_dict_pylist_from_row_partitions(row_partitions[1:], row_partitions[0].row_splits()[-1])
        splits = row_partitions[0].row_splits()
        return [values[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]