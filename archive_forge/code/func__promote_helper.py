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
def _promote_helper(self, source_path, new_parent_path):
    """Creates a promoted field without adding it to the structure.

    Args:
      source_path: the source path in the structured tensor.
      new_parent_path: the new parent path. Must be a prefix of source_path.

    Returns:
      a composite tensor of source_path promoted.
    Raises:
      ValueError: if the shape of the field is unknown and the right strategy
      cannot be determined.
    """
    current_field = self.field_value(source_path)
    new_parent_rank = self.field_value(new_parent_path).rank
    parent_rank = self.field_value(source_path[:-1]).rank
    if new_parent_rank == parent_rank:
        return current_field
    current_field_rank = current_field.shape.rank
    if current_field_rank is None:
        raise ValueError('Cannot determine if dimensions should be merged.')
    inner_dim = min(parent_rank, current_field_rank - 1)
    if inner_dim <= new_parent_rank:
        return current_field
    return _merge_dims_generic(current_field, new_parent_rank, inner_dim)