import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import index_lookup
from tensorflow.python.util.tf_export import keras_export
def _tensor_vocab_to_numpy(self, vocabulary):
    vocabulary = vocabulary.numpy()
    return np.array([tf.compat.as_text(x, self.encoding) for x in vocabulary])