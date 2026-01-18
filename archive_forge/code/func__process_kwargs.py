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
def _process_kwargs(self, kwargs):
    kwargs.pop('is_legacy_optimizer', None)
    lr = kwargs.pop('lr', None)
    if lr:
        logging.warning(f'`lr` is deprecated in Keras optimizer, please use `learning_rate` or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.{self.__class__.__name__}.')
    legacy_kwargs = {'decay', 'gradient_aggregator', 'gradient_transformers'}
    for k in kwargs:
        if k in legacy_kwargs:
            raise ValueError(f'{k} is deprecated in the new Keras optimizer, please check the docstring for valid arguments, or use the legacy optimizer, e.g., tf.keras.optimizers.legacy.{self.__class__.__name__}.')
        else:
            raise TypeError(f'{k} is not a valid argument, kwargs should be empty  for `optimizer_experimental.Optimizer`.')