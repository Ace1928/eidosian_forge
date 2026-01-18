import abc
import platform
import re
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src import initializers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def _build_learning_rate(self, learning_rate):
    if not self._mesh:
        return super()._build_learning_rate(learning_rate)
    variable_creation = tf.experimental.dtensor.DVariable
    init_value_convert_fn = lambda x: tf.experimental.dtensor.copy_to_mesh(x, tf.experimental.dtensor.Layout.replicated(self._mesh, rank=0))
    if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
        current_learning_rate = tf.convert_to_tensor(learning_rate(self.iterations))
        current_learning_rate = init_value_convert_fn(current_learning_rate)
        self._current_learning_rate = variable_creation(current_learning_rate, name='learning_rate', dtype=tf.float32)
        return learning_rate
    init_val = init_value_convert_fn(tf.constant(learning_rate, dtype=tf.float32))
    return variable_creation(init_val, name='learning_rate', dtype=backend.floatx(), trainable=False)