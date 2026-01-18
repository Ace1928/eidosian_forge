import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _compute_dx_and_dy(x, y, y_shape):
    """Returns a node to compute gradient of y wrt x."""
    with x.graph.as_default():
        dy_orig = constant_op.constant(1.0, shape=y_shape, dtype=y.dtype)
        dy = array_ops.identity(dy_orig)
    grads = gradients.gradients(y, x, dy)
    assert len(grads) == 1
    return (grads[0], dy_orig)