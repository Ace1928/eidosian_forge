import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def _renorm_correction_and_moments(self, mean, variance, training, inputs_size):
    """Returns the correction and update values for renorm."""
    stddev = tf.sqrt(variance + self.epsilon)
    renorm_mean = self.renorm_mean
    renorm_stddev = tf.maximum(self.renorm_stddev, tf.sqrt(self.epsilon))
    r = stddev / renorm_stddev
    d = (mean - renorm_mean) / renorm_stddev
    with tf.control_dependencies([r, d]):
        mean = tf.identity(mean)
        stddev = tf.identity(stddev)
    rmin, rmax, dmax = [self.renorm_clipping.get(key) for key in ['rmin', 'rmax', 'dmax']]
    if rmin is not None:
        r = tf.maximum(r, rmin)
    if rmax is not None:
        r = tf.minimum(r, rmax)
    if dmax is not None:
        d = tf.maximum(d, -dmax)
        d = tf.minimum(d, dmax)
    r = control_flow_util.smart_cond(training, lambda: r, lambda: tf.ones_like(r))
    d = control_flow_util.smart_cond(training, lambda: d, lambda: tf.zeros_like(d))

    def _update_renorm_variable(var, value, inputs_size):
        """Updates a moving average and weight, returns the unbiased
            value."""
        value = tf.identity(value)

        def _do_update():
            """Updates the var, returns the updated value."""
            new_var = self._assign_moving_average(var, value, self.renorm_momentum, inputs_size)
            return new_var

        def _fake_update():
            return tf.identity(var)
        return control_flow_util.smart_cond(training, _do_update, _fake_update)
    update_new_mean = _update_renorm_variable(self.renorm_mean, mean, inputs_size)
    update_new_stddev = _update_renorm_variable(self.renorm_stddev, stddev, inputs_size)
    with tf.control_dependencies([update_new_mean, update_new_stddev]):
        out_mean = tf.identity(mean)
        out_variance = tf.identity(variance)
    return (r, d, out_mean, out_variance)