import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights tensors.

  Also available via the shortcut function
  `tf.keras.initializers.variance_scaling`.

  With `distribution="truncated_normal" or "untruncated_normal"`, samples are
  drawn from a truncated/untruncated normal distribution with a mean of zero and
  a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`,
  where `n` is:

  - number of input units in the weight tensor, if `mode="fan_in"`
  - number of output units, if `mode="fan_out"`
  - average of the numbers of input and output units, if `mode="fan_avg"`

  With `distribution="uniform"`, samples are drawn from a uniform distribution
  within `[-limit, limit]`, where `limit = sqrt(3 * scale / n)`.

  Examples:

  >>> # Standalone usage:
  >>> initializer = tf.keras.initializers.VarianceScaling(
  ... scale=0.1, mode='fan_in', distribution='uniform')
  >>> values = initializer(shape=(2, 2))

  >>> # Usage in a Keras layer:
  >>> initializer = tf.keras.initializers.VarianceScaling(
  ... scale=0.1, mode='fan_in', distribution='uniform')
  >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

  Args:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "truncated_normal",
      "untruncated_normal" and  "uniform".
    seed: A Python integer. An initializer created with a given seed will
      always produce the same random tensor for a given shape and dtype.
  """

    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None):
        if scale <= 0.0:
            raise ValueError('`scale` must be positive float.')
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument:', mode)
        distribution = distribution.lower()
        if distribution == 'normal':
            distribution = 'truncated_normal'
        if distribution not in {'uniform', 'truncated_normal', 'untruncated_normal'}:
            raise ValueError('Invalid `distribution` argument:', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        self._random_generator = _RandomGenerator(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only floating point types are
        supported. If not specified, `tf.keras.backend.floatx()` is used, which
        default to `float32` unless you configured it otherwise (via
        `tf.keras.backend.set_floatx(float_dtype)`)
      **kwargs: Additional keyword arguments.
    """
        _validate_kwargs(self.__class__.__name__, kwargs)
        dtype = _assert_float_dtype(_get_dtype(dtype))
        scale = self.scale
        fan_in, fan_out = _compute_fans(shape)
        if _PARTITION_SHAPE in kwargs:
            shape = kwargs[_PARTITION_SHAPE]
        if self.mode == 'fan_in':
            scale /= max(1.0, fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
        if self.distribution == 'truncated_normal':
            stddev = math.sqrt(scale) / 0.8796256610342398
            return self._random_generator.truncated_normal(shape, 0.0, stddev, dtype)
        elif self.distribution == 'untruncated_normal':
            stddev = math.sqrt(scale)
            return self._random_generator.random_normal(shape, 0.0, stddev, dtype)
        else:
            limit = math.sqrt(3.0 * scale)
            return self._random_generator.random_uniform(shape, -limit, limit, dtype)

    def get_config(self):
        return {'scale': self.scale, 'mode': self.mode, 'distribution': self.distribution, 'seed': self.seed}