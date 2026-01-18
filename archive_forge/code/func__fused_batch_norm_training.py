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
def _fused_batch_norm_training():
    return tf.compat.v1.nn.fused_batch_norm(inputs, gamma, beta, mean=self.moving_mean, variance=_maybe_add_or_remove_bessels_correction(self.moving_variance, remove=False), epsilon=self.epsilon, is_training=True, data_format=self._data_format, exponential_avg_factor=exponential_avg_factor)