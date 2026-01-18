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
def _matrix_conv(self, m1, m2):
    """Matrix convolution.

    Args:
      m1: is a k x k x k  dictionary, each element is a n x n matrix.
      m2: is a l x l x l dictionary, each element is a n x n matrix.

    Returns:
      (k + l - 1) x (k + l - 1) x (k + l - 1) dictionary each
      element is a n x n matrix.
    Raises:
      ValueError: if the entries of m1 and m2 are of different dimensions.
    """
    n = m1[0, 0, 0].shape.as_list()[0]
    if n != m2[0, 0, 0].shape.as_list()[0]:
        raise ValueError(f'The entries in matrices m1 and m2 must have the same dimensions. Received m1[0, 0, 0].shape={m1[0, 0, 0].shape} and m2[0, 0, 0].shape={m2[0, 0, 0].shape}.')
    k = int(np.cbrt(len(m1)))
    l = int(np.cbrt(len(m2)))
    result = {}
    size = k + l - 1
    for i in range(size):
        for j in range(size):
            for r in range(size):
                result[i, j, r] = array_ops.zeros([n, n], self.dtype)
                for index1 in range(min(k, i + 1)):
                    for index2 in range(min(k, j + 1)):
                        for index3 in range(min(k, r + 1)):
                            if i - index1 < l and j - index2 < l and (r - index3 < l):
                                result[i, j, r] += math_ops.matmul(m1[index1, index2, index3], m2[i - index1, j - index2, r - index3])
    return result