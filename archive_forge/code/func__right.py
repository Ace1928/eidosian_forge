from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def _right(operator):
    """Right-handed version of an operator: swap args x and y."""
    return tf_decorator.make_decorator(operator, lambda y, x: operator(x, y))