import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _check_num_rows_possibly_add_asserts(self):
    """Static check of init arg `num_rows`, possibly add asserts."""
    if self._assert_proper_shapes:
        self._num_rows = control_flow_ops.with_dependencies([check_ops.assert_rank(self._num_rows, 0, message='Argument num_rows must be a 0-D Tensor.'), check_ops.assert_non_negative(self._num_rows, message='Argument num_rows must be non-negative.')], self._num_rows)
    if not self._num_rows.dtype.is_integer:
        raise TypeError('Argument num_rows must be integer type.  Found: %s' % self._num_rows)
    num_rows_static = self._num_rows_static
    if num_rows_static is None:
        return
    if num_rows_static.ndim != 0:
        raise ValueError('Argument num_rows must be a 0-D Tensor.  Found: %s' % num_rows_static)
    if num_rows_static < 0:
        raise ValueError('Argument num_rows must be non-negative.  Found: %s' % num_rows_static)