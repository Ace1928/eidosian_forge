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
class ConvolutionOrthogonal(Initializer):
    """Initializer that generates orthogonal kernel for ConvNets.

  Base class used to construct 1D, 2D and 3D orthogonal kernels for convolution.

  Args:
    gain: multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  """

    def __init__(self, gain=1.0, seed=None, dtype=dtypes.float32):
        self.gain = gain
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        raise NotImplementedError

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed, 'dtype': self.dtype.name}

    def _orthogonal_matrix(self, n):
        """Construct an n x n orthogonal matrix.

    Args:
      n: Dimension.

    Returns:
      A n x n orthogonal matrix.
    """
        a = random_ops.random_normal([n, n], dtype=self.dtype, seed=self.seed)
        if self.seed:
            self.seed += 1
        q, r = gen_linalg_ops.qr(a)
        d = array_ops.diag_part(r)
        q *= math_ops.sign(d)
        return q

    def _symmetric_projection(self, n):
        """Compute a n x n symmetric projection matrix.

    Args:
      n: Dimension.

    Returns:
      A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
    """
        q = self._orthogonal_matrix(n)
        mask = math_ops.cast(random_ops.random_normal([n], seed=self.seed) > 0, self.dtype)
        if self.seed:
            self.seed += 1
        c = math_ops.multiply(q, mask)
        return math_ops.matmul(c, array_ops.matrix_transpose(c))