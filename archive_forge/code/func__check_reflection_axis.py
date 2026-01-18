from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _check_reflection_axis(self, reflection_axis):
    """Static check of reflection_axis."""
    if reflection_axis.shape.ndims is not None and reflection_axis.shape.ndims < 1:
        raise ValueError('Argument reflection_axis must have at least 1 dimension.  Found: %s' % reflection_axis)