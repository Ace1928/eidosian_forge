import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from tensorflow.python.util.tf_export import keras_export
def _standardize_inputs(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    if inputs.dtype != self.compute_dtype:
        inputs = tf.cast(inputs, self.compute_dtype)
    return inputs