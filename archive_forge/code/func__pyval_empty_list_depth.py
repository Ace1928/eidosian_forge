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
def _pyval_empty_list_depth(pyval):
    """Find the max depth for nested empty lists.

  Args:
    pyval: A nested python list.

  Returns:
    The maximum depth of empty lists in `pyval`, or None if `pyval` contains
    anything other than nested empty lists.
  """
    if isinstance(pyval, list):
        if not pyval:
            return 1
        depths = [_pyval_empty_list_depth(v) for v in pyval]
        if any((depth is None for depth in depths)):
            return None
        else:
            return max(depths) + 1
    else:
        return None