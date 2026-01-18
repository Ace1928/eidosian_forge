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
def _get_uv_as_tensors(self):
    """Get (self.u, self.v) as tensors (in case they were refs)."""
    u = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.u)
    if self.v is self.u:
        v = u
    else:
        v = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.v)
    return (u, v)