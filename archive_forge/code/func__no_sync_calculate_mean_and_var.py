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
def _no_sync_calculate_mean_and_var(self, inputs, reduction_axes, keep_dims, mask=None):
    if mask is None:
        return tf.nn.moments(inputs, reduction_axes, keepdims=keep_dims)
    else:
        mask_weights = tf.cast(mask, self.compute_dtype, name='mask_weights')
        mask_weights = tf.expand_dims(mask_weights, axis=-1, name='mask_weights_broadcasted')
        return tf.nn.weighted_moments(inputs, axes=reduction_axes, frequency_weights=mask_weights, keepdims=keep_dims)