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
def _lookup_table_from_tokens(self, tokens):
    with tf.init_scope():
        token_start = self._token_start_index()
        token_end = token_start + tf.size(tokens)
        indices_dtype = self._key_dtype if self.invert else self._value_dtype
        indices = tf.range(token_start, token_end, dtype=indices_dtype)
        keys, values = (indices, tokens) if self.invert else (tokens, indices)
        initializer = tf.lookup.KeyValueTensorInitializer(keys, values, self._key_dtype, self._value_dtype)
        return tf.lookup.StaticHashTable(initializer, self._default_value)