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
def _update_step(self, gradient, variable):
    if getattr(variable, '_unique_id', None) is None:
        return
    if self._var_key(variable) not in self._index_dict:
        raise KeyError(f'The optimizer cannot recognize variable {variable.name}. This usually means you are trying to call the optimizer to update different parts of the model separately. Please call `optimizer.build(variables)` with the full list of trainable variables before the training loop or use legacy optimizer `tf.keras.optimizers.legacy.{self.__class__.__name__}.')
    self.update_step(gradient, variable)