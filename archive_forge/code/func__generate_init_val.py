import math
import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils
from keras.src.saving import serialization_lib
from tensorflow.python.util.tf_export import keras_export
def _generate_init_val(self, shape, dtype):
    initializer = tf.eye(*shape, dtype=dtype)
    return self.gain * initializer