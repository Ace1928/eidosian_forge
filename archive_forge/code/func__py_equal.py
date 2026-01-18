from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def _py_equal(a, b):
    """Overload of "equal" that falls back to Python's default implementation."""
    return a == b