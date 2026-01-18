import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.engine import base_layer
from keras.src.engine import input_spec
from tensorflow.python.util.tf_export import keras_export
def _get_default_scale(initializer, input_dim):
    if isinstance(initializer, str) and initializer.lower() == 'gaussian':
        return np.sqrt(input_dim / 2.0)
    return 1.0