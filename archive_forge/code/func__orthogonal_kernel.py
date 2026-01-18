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
def _orthogonal_kernel(self, ksize, cin, cout):
    """Construct orthogonal kernel for convolution.

    Args:
      ksize: Kernel size.
      cin: Number of input channels.
      cout: Number of output channels.

    Returns:
      An [ksize, ksize, ksize, cin, cout] orthogonal kernel.
    Raises:
      ValueError: If cin > cout.
    """
    if cin > cout:
        raise ValueError(f'The number of input channels (cin={cin}) cannot exceed the number of output channels (cout={cout}).')
    orth = self._orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
        return array_ops.expand_dims(array_ops.expand_dims(array_ops.expand_dims(orth, 0), 0), 0)
    p = self._block_orth(self._symmetric_projection(cout), self._symmetric_projection(cout), self._symmetric_projection(cout))
    for _ in range(ksize - 2):
        temp = self._block_orth(self._symmetric_projection(cout), self._symmetric_projection(cout), self._symmetric_projection(cout))
        p = self._matrix_conv(p, temp)
    for i in range(ksize):
        for j in range(ksize):
            for k in range(ksize):
                p[i, j, k] = math_ops.matmul(orth, p[i, j, k])
    return self._dict_to_tensor(p, ksize, ksize, ksize)