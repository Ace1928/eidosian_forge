from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
def _CustomReciprocal(x):
    """Wrapper function around `math_ops.div_no_nan()` to perform a "safe" reciprocal incase the input is zero. Avoids divide by zero and NaNs.

  Input:
    x -> input tensor to be reciprocat-ed.
  Returns:
    x_reciprocal -> reciprocal of x without NaNs.
  """
    return math_ops.div_no_nan(math_ops.cast(1.0, x.dtype), x)