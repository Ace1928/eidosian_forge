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
def _lookup_table_from_file(self, filename):
    if self.invert:
        key_index = tf.lookup.TextFileIndex.LINE_NUMBER
        value_index = tf.lookup.TextFileIndex.WHOLE_LINE
    else:
        key_index = tf.lookup.TextFileIndex.WHOLE_LINE
        value_index = tf.lookup.TextFileIndex.LINE_NUMBER
    with tf.init_scope():
        initializer = tf.lookup.TextFileInitializer(filename=filename, key_dtype=self._key_dtype, key_index=key_index, value_dtype=self._value_dtype, value_index=value_index, value_index_offset=self._token_start_index())
        return tf.lookup.StaticHashTable(initializer, self._default_value)