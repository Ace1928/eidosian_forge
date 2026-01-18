import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.dtensor import utils as dtensor_utils
from keras.src.metrics import base_metric
def _add_zeros_weight(name):
    return self.add_weight(name, shape=init_shape, initializer='zeros', dtype=self.dtype)