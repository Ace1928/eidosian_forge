import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import gamma
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distributions.Exponential'])
class Exponential(gamma.Gamma):
    """Exponential distribution.

  The Exponential distribution is parameterized by an event `rate` parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; lambda, x > 0) = exp(-lambda x) / Z
  Z = 1 / lambda
  ```

  where `rate = lambda` and `Z` is the normalizaing constant.

  The Exponential distribution is a special case of the Gamma distribution,
  i.e.,

  ```python
  Exponential(rate) = Gamma(concentration=1., rate)
  ```

  The Exponential distribution uses a `rate` parameter, or "inverse scale",
  which can be intuited as,

  ```none
  X ~ Exponential(rate=1)
  Y = X / rate
  ```

  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, rate, validate_args=False, allow_nan_stats=True, name='Exponential'):
        """Construct Exponential distribution with parameter `rate`.

    Args:
      rate: Floating point tensor, equivalent to `1 / mean`. Must contain only
        positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
        parameters = dict(locals())
        with ops.name_scope(name, values=[rate]) as name:
            self._rate = ops.convert_to_tensor(rate, name='rate')
        super(Exponential, self).__init__(concentration=array_ops.ones([], dtype=self._rate.dtype), rate=self._rate, allow_nan_stats=allow_nan_stats, validate_args=validate_args, name=name)
        self._parameters = parameters
        self._graph_parents += [self._rate]

    @staticmethod
    def _param_shapes(sample_shape):
        return {'rate': ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

    @property
    def rate(self):
        return self._rate

    def _log_survival_function(self, value):
        return self._log_prob(value) - math_ops.log(self._rate)

    def _sample_n(self, n, seed=None):
        shape = array_ops.concat([[n], array_ops.shape(self._rate)], 0)
        sampled = random_ops.random_uniform(shape, minval=np.finfo(self.dtype.as_numpy_dtype).tiny, maxval=1.0, seed=seed, dtype=self.dtype)
        return -math_ops.log(sampled) / self._rate