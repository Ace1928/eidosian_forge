from tensorflow.python.framework import constant_op
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation
def _forward_log_det_jacobian(self, x):
    return constant_op.constant(0.0, dtype=x.dtype)