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
def _maybe_add_or_remove_bessels_correction(variance, remove=True):
    """Add or remove Bessel's correction."""
    if self._bessels_correction_test_only:
        return variance
    sample_size = tf.cast(tf.size(inputs) / tf.size(variance), variance.dtype)
    if remove:
        factor = (sample_size - tf.cast(1.0, variance.dtype)) / sample_size
    else:
        factor = sample_size / (sample_size - tf.cast(1.0, variance.dtype))
    return variance * factor