from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _check_shapes(self):
    """Static check that shapes are compatible."""
    uv_shape = array_ops.broadcast_static_shape(self.u.shape, self.v.shape)
    batch_shape = array_ops.broadcast_static_shape(self.base_operator.batch_shape, uv_shape[:-2])
    tensor_shape.Dimension(self.base_operator.domain_dimension).assert_is_compatible_with(uv_shape[-2])
    if self._diag_update is not None:
        tensor_shape.dimension_at_index(uv_shape, -1).assert_is_compatible_with(self._diag_update.shape[-1])
        array_ops.broadcast_static_shape(batch_shape, self._diag_update.shape[:-1])