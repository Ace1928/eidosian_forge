import math
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_arg_values
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _dict_to_tensor(self, x, k1, k2, k3):
    """Convert a dictionary to a tensor.

    Args:
      x: A k1 * k2 dictionary.
      k1: First dimension of x.
      k2: Second dimension of x.
      k3: Third dimension of x.

    Returns:
      A k1 * k2 * k3 tensor.
    """
    return array_ops_stack.stack([array_ops_stack.stack([array_ops_stack.stack([x[i, j, k] for k in range(k3)]) for j in range(k2)]) for i in range(k1)])