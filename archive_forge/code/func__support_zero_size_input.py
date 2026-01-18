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
def _support_zero_size_input(self):
    if not tf.distribute.has_strategy():
        return False
    strategy = tf.distribute.get_strategy()
    return getattr(strategy.extended, 'enable_partial_batch_handling', getattr(strategy.extended, 'experimental_enable_get_next_as_optional', False))