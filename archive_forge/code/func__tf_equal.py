from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops
def _tf_equal(a, b):
    """Overload of "equal" for Tensors."""
    return gen_math_ops.equal(a, b)