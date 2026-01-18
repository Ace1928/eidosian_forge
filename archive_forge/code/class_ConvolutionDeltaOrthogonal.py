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
class ConvolutionDeltaOrthogonal(Initializer):
    """Initializer that generates a delta orthogonal kernel for ConvNets.

  The shape of the tensor must have length 3, 4 or 5. The number of input
  filters must not exceed the number of output filters. The center pixels of the
  tensor form an orthogonal matrix. Other pixels are set to be zero. See
  algorithm 2 in (Xiao et al., 2018).


  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
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
        if dtype is None:
            dtype = self.dtype
        if len(shape) < 3 or len(shape) > 5:
            raise ValueError(f'The tensor to initialize, specified by argument `shape` must be at least three-dimensional and at most five-dimensional. Received shape={shape}')
        if shape[-2] > shape[-1]:
            raise ValueError(f'In_filters, specified by shape[-2]={shape[-2]} cannot be greater than out_filters, specified by shape[-1]={shape[-1]}.')
        a = random_ops.random_normal([shape[-1], shape[-1]], dtype=dtype, seed=self.seed)
        q, r = gen_linalg_ops.qr(a, full_matrices=False)
        d = array_ops.diag_part(r)
        q *= math_ops.sign(d)
        q = q[:shape[-2], :]
        q *= math_ops.cast(self.gain, dtype=dtype)
        if len(shape) == 3:
            weight = array_ops.scatter_nd([[(shape[0] - 1) // 2]], array_ops.expand_dims(q, 0), shape)
        elif len(shape) == 4:
            weight = array_ops.scatter_nd([[(shape[0] - 1) // 2, (shape[1] - 1) // 2]], array_ops.expand_dims(q, 0), shape)
        else:
            weight = array_ops.scatter_nd([[(shape[0] - 1) // 2, (shape[1] - 1) // 2, (shape[2] - 1) // 2]], array_ops.expand_dims(q, 0), shape)
        return weight

    def get_config(self):
        return {'gain': self.gain, 'seed': self.seed, 'dtype': self.dtype.name}