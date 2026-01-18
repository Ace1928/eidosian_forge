from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def _py_lazy_or(cond, b):
    """Lazy-eval equivalent of "or" in Python."""
    return cond or b()