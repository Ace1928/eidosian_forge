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
@tf_export(v1=['initializers.uniform_unit_scaling', 'uniform_unit_scaling_initializer'])
@deprecation.deprecated_endpoints('uniform_unit_scaling_initializer', 'initializers.uniform_unit_scaling')
class UniformUnitScaling(Initializer):
    """Initializer that generates tensors without scaling variance.

  When initializing a deep network, it is in principle advantageous to keep
  the scale of the input variance constant, so it does not explode or diminish
  by reaching the final layer. If the input is `x` and the operation `x * W`,
  and we want to initialize `W` uniformly at random, we need to pick `W` from

      [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

  to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
  A similar calculation for convolutional networks gives an analogous result
  with `dim` equal to the product of the first 3 dimensions.  When
  nonlinearities are present, we need to multiply this by a constant `factor`.
  See (Sussillo et al., 2014) for deeper motivation, experiments
  and the calculation of constants. In section 2.3 there, the constants were
  numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

  Args:
    factor: Float.  A multiplicative factor by which the values will be scaled.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
      ([pdf](http://arxiv.org/pdf/1412.6558.pdf))
  """

    @deprecated_args(None, 'Call initializer instance with the dtype argument instead of passing it to the constructor', 'dtype')
    @deprecated(None, 'Use tf.initializers.variance_scaling instead with distribution=uniform to get equivalent behavior.')
    def __init__(self, factor=1.0, seed=None, dtype=dtypes.float32):
        self.factor = factor
        self.seed = seed
        self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        scale_shape = shape
        if partition_info is not None:
            scale_shape = partition_info.full_shape
        input_size = 1.0
        for dim in scale_shape[:-1]:
            input_size *= float(dim)
        input_size = max(input_size, 1.0)
        max_val = math.sqrt(3 / input_size) * self.factor
        return random_ops.random_uniform(shape, -max_val, max_val, dtype, seed=self.seed)

    def get_config(self):
        return {'factor': self.factor, 'seed': self.seed, 'dtype': self.dtype.name}