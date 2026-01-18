from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def _tf_not(a):
    """Implementation of the "not_" operator for TensorFlow."""
    return gen_math_ops.logical_not(a)