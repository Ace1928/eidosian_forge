import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distributions.Gamma'])
class Gamma(distribution.Distribution):
    """Gamma distribution.

  The Gamma distribution is defined over positive real numbers using
  parameters `concentration` (aka "alpha") and `rate` (aka "beta").

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta, x > 0) = x**(alpha - 1) exp(-x beta) / Z
  Z = Gamma(alpha) beta**(-alpha)
  ```

  where:

  * `concentration = alpha`, `alpha > 0`,
  * `rate = beta`, `beta > 0`,
  * `Z` is the normalizing constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The cumulative density function (cdf) is,

  ```none
  cdf(x; alpha, beta, x > 0) = GammaInc(alpha, beta x) / Gamma(alpha)
  ```

  where `GammaInc` is the [lower incomplete Gamma function](
  https://en.wikipedia.org/wiki/Incomplete_gamma_function).

  The parameters can be intuited via their relationship to mean and stddev,

  ```none
  concentration = alpha = (mean / stddev)**2
  rate = beta = mean / stddev**2 = concentration / mean
  ```

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  Warning: The samples of this distribution are always non-negative. However,
  the samples that are smaller than `np.finfo(dtype).tiny` are rounded
  to this value, so it appears more often than it should.
  This should only be noticeable when the `concentration` is very small, or the
  `rate` is very large. See note in `tf.random.gamma` docstring.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in
  (Figurnov et al., 2018).

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dist = tfd.Gamma(concentration=3.0, rate=2.0)
  dist2 = tfd.Gamma(concentration=[3.0, 4.0], rate=[2.0, 3.0])
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  concentration = tf.constant(3.0)
  rate = tf.constant(2.0)
  dist = tfd.Gamma(concentration, rate)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [concentration, rate])
  ```

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf](http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, concentration, rate, validate_args=False, allow_nan_stats=True, name='Gamma'):
        """Construct Gamma with `concentration` and `rate` parameters.

    The parameters `concentration` and `rate` must be shaped in a way that
    supports broadcasting (e.g. `concentration + rate` is a valid operation).

    Args:
      concentration: Floating point tensor, the concentration params of the
        distribution(s). Must contain only positive values.
      rate: Floating point tensor, the inverse scale params of the
        distribution(s). Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `concentration` and `rate` are different dtypes.
    """
        parameters = dict(locals())
        with ops.name_scope(name, values=[concentration, rate]) as name:
            with ops.control_dependencies([check_ops.assert_positive(concentration), check_ops.assert_positive(rate)] if validate_args else []):
                self._concentration = array_ops.identity(concentration, name='concentration')
                self._rate = array_ops.identity(rate, name='rate')
                check_ops.assert_same_float_dtype([self._concentration, self._rate])
        super(Gamma, self).__init__(dtype=self._concentration.dtype, validate_args=validate_args, allow_nan_stats=allow_nan_stats, reparameterization_type=distribution.FULLY_REPARAMETERIZED, parameters=parameters, graph_parents=[self._concentration, self._rate], name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(zip(('concentration', 'rate'), [ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)] * 2))

    @property
    def concentration(self):
        """Concentration parameter."""
        return self._concentration

    @property
    def rate(self):
        """Rate parameter."""
        return self._rate

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(array_ops.shape(self.concentration), array_ops.shape(self.rate))

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(self.concentration.get_shape(), self.rate.get_shape())

    def _event_shape_tensor(self):
        return constant_op.constant([], dtype=dtypes.int32)

    def _event_shape(self):
        return tensor_shape.TensorShape([])

    @distribution_util.AppendDocstring('Note: See `tf.random.gamma` docstring for sampling details and\n      caveats.')
    def _sample_n(self, n, seed=None):
        return random_ops.random_gamma(shape=[n], alpha=self.concentration, beta=self.rate, dtype=self.dtype, seed=seed)

    def _log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _cdf(self, x):
        x = self._maybe_assert_valid_sample(x)
        return math_ops.igamma(self.concentration, self.rate * x)

    def _log_unnormalized_prob(self, x):
        x = self._maybe_assert_valid_sample(x)
        return math_ops.xlogy(self.concentration - 1.0, x) - self.rate * x

    def _log_normalization(self):
        return math_ops.lgamma(self.concentration) - self.concentration * math_ops.log(self.rate)

    def _entropy(self):
        return self.concentration - math_ops.log(self.rate) + math_ops.lgamma(self.concentration) + (1.0 - self.concentration) * math_ops.digamma(self.concentration)

    def _mean(self):
        return self.concentration / self.rate

    def _variance(self):
        return self.concentration / math_ops.square(self.rate)

    def _stddev(self):
        return math_ops.sqrt(self.concentration) / self.rate

    @distribution_util.AppendDocstring('The mode of a gamma distribution is `(shape - 1) / rate` when\n      `shape > 1`, and `NaN` otherwise. If `self.allow_nan_stats` is `False`,\n      an exception will be raised rather than returning `NaN`.')
    def _mode(self):
        mode = (self.concentration - 1.0) / self.rate
        if self.allow_nan_stats:
            nan = array_ops.fill(self.batch_shape_tensor(), np.array(np.nan, dtype=self.dtype.as_numpy_dtype()), name='nan')
            return array_ops.where_v2(self.concentration > 1.0, mode, nan)
        else:
            return control_flow_ops.with_dependencies([check_ops.assert_less(array_ops.ones([], self.dtype), self.concentration, message='mode not defined when any concentration <= 1')], mode)

    def _maybe_assert_valid_sample(self, x):
        check_ops.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
        if not self.validate_args:
            return x
        return control_flow_ops.with_dependencies([check_ops.assert_positive(x)], x)