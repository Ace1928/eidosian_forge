import math
import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils
from keras.src.saving import serialization_lib
from tensorflow.python.util.tf_export import keras_export
def _warn_reuse(self):
    if getattr(self, '_used', False):
        if getattr(self, 'seed', None) is None:
            warnings.warn(f'The initializer {self.__class__.__name__} is unseeded and being called multiple times, which will return identical values each time (even if the initializer is unseeded). Please update your code to provide a seed to the initializer, or avoid using the same initializer instance more than once.')
    else:
        self._used = True