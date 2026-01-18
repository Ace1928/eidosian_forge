from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _validate_non_singular(self, is_non_singular):
    if all((op.is_non_singular for op in self._diagonal_operators)):
        if is_non_singular is False:
            raise ValueError(f'A blockwise lower-triangular operator with non-singular operators on the main diagonal is always non-singular. Expected argument `is_non_singular` to be True. Received: {is_non_singular}.')
        return True
    if any((op.is_non_singular is False for op in self._diagonal_operators)):
        if is_non_singular is True:
            raise ValueError(f'A blockwise lower-triangular operator with a singular operator on the main diagonal is always singular. Expected argument `is_non_singular` to be True. Received: {is_non_singular}.')
        return False