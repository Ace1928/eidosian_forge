from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _get_tril(self):
    """Gets the `tril` kwarg, with upper part zero-d out."""
    return array_ops.matrix_band_part(self._tril, -1, 0)