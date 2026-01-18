from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import distributions
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import model
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import PredictionFeatures
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def _compute_time_features(self, time):
    """Compute some features on the time value."""
    batch_size = tf.compat.v1.shape(time)[0]
    num_periods = len(self._periodicities)
    periods = tf.constant(self._periodicities, shape=[1, 1, num_periods, 1], dtype=time.dtype)
    time = tf.reshape(time, [batch_size, -1, 1, 1])
    window_offset = time / self._periodicities
    mod = tf.cast(time % periods, self.dtype) * self._buckets / tf.cast(periods, self.dtype)
    intervals = tf.reshape(tf.range(self._buckets, dtype=self.dtype), [1, 1, 1, self._buckets])
    mod = tf.nn.relu(mod - intervals)
    mod = tf.where(mod < 1.0, mod, tf.compat.v1.zeros_like(mod))
    return (window_offset, mod)