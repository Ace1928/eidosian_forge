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
def _merge_nrows(nrows, static_nrows, value, dtype, validate):
    """Merges `nrows` with `nrows(value)`.

  Checks that `value` has the expected number of rows (`nrows`), and returns
  `nrows`.  If `validate` is true, then add validation ops that check that
  the `nrows` values match.

  Args:
    nrows: scalar integer Tensor.
    static_nrows: tf.Dimension: static value of nrows, if known.
    value: Tensor or RaggedTensor or StructuredTensor
    dtype: dtype for `nrows`.
    validate: bool -- whether to add validation ops.

  Returns:
    A tuple `(nrows, static_nrows)`.
  """
    static_value_nrows = tensor_shape.dimension_at_index(value.shape, 0)
    if isinstance(value, tensor.Tensor):
        value_nrows = array_ops.shape(value, out_type=dtype)[0]
    else:
        value_nrows = value.nrows()
    if nrows is None:
        nrows = value_nrows
    elif static_value_nrows.value is not None and static_nrows.value is not None:
        if not static_value_nrows.is_compatible_with(static_nrows):
            raise ValueError('fields have incompatible nrows')
        nrows = value_nrows
    elif validate:
        nrows = control_flow_ops.with_dependencies([check_ops.assert_equal(nrows, value_nrows, message='fields have incompatible nrows')], nrows)
    return (nrows, static_nrows._merge_with(static_value_nrows))