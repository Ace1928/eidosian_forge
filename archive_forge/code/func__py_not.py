from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def _py_not(a):
    """Default Python implementation of the "not_" operator."""
    return not a