import collections
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
class VocabWeightHandler(base_layer_utils.TrackableWeightHandler):
    """Adds the vocabulary as a layer weight during serialization."""

    def __init__(self, lookup_layer):
        self._layer = lookup_layer
        self._dtype = lookup_layer.vocabulary_dtype
        self._distribute_strategy = tf.distribute.get_strategy()

    @property
    def num_tensors(self):
        return 1

    def set_weights(self, weights):
        tokens = tf.convert_to_tensor(weights[0], self._dtype)
        self._layer.lookup_table = self._layer._lookup_table_from_tokens(tokens)

    def get_tensors(self):
        tokens = self._layer.get_vocabulary(include_special_tokens=False)
        tokens = tf.convert_to_tensor(tokens, self._dtype)
        return [tokens]