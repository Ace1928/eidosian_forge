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
def _compute_current_learning_rate(self):
    if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
        if hasattr(self, '_current_learning_rate'):
            self._current_learning_rate.assign(self._learning_rate(self.iterations))
        else:
            current_learning_rate = tf.convert_to_tensor(self._learning_rate(self.iterations))
            self._current_learning_rate = tf.Variable(current_learning_rate, name='current_learning_rate', dtype=current_learning_rate.dtype, trainable=False)