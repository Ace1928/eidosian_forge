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
def _merge_dims_generic(source, outer, inner):
    """Merges outer_axis...inner_axis into a single dimension.

  If outer == inner, this is a NOOP. If inner < outer, then this fials.
  If inner >= source.shape.rank, then the behavior is undefined.

  Args:
    source: a tensor, ragged tensor, or structured tensor.
    outer: a python int, indicating the first dimension to compress (must be
      nonnegative).
    inner: a python int, indicating the first dimension to keep (of the tail)
      (must be nonnegative).

  Returns:
    source with outer_axis...inner_axis merged into a single dimension.

  """
    if isinstance(source, StructuredTensor):
        return source.merge_dims(outer, inner)
    else:
        return ragged_tensor.merge_dims(source, outer, inner)