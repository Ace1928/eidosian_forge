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
def _set_diag_operators(self, diag_update, is_diag_update_positive):
    """Set attributes self._diag_update and self._diag_operator."""
    if diag_update is not None:
        self._diag_operator = linear_operator_diag.LinearOperatorDiag(self._diag_update, is_positive_definite=is_diag_update_positive)
    else:
        if tensor_shape.dimension_value(self.u.shape[-1]) is not None:
            r = tensor_shape.dimension_value(self.u.shape[-1])
        else:
            r = array_ops.shape(self.u)[-1]
        self._diag_operator = linear_operator_identity.LinearOperatorIdentity(num_rows=r, dtype=self.dtype)