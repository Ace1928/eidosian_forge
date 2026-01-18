import os
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src import callbacks as callbacks_lib
from keras.src.engine import sequential
from keras.src.layers import core as core_layers
from keras.src.layers.preprocessing import string_lookup
from keras.src.optimizers.legacy import gradient_descent
from keras.src.utils import dataset_creator
from tensorflow.python.platform import tf_logging as logging
def _get_dataset_fn(self, use_lookup_layer):
    if use_lookup_layer:
        filepath = os.path.join(self.get_temp_dir(), 'vocab')
        with open(filepath, 'w') as f:
            f.write('\n'.join(['earth', 'wind', 'and', 'fire']))

        def dataset_fn(input_context):
            del input_context
            lookup_layer = string_lookup.StringLookup(num_oov_indices=1, vocabulary=filepath)
            x = np.array([['earth', 'wind', 'and', 'fire'], ['fire', 'and', 'earth', 'michigan']])
            y = np.array([0, 1])
            map_fn = lambda x, y: (lookup_layer(x), y)
            return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat().batch(2).map(map_fn)
    else:

        def dataset_fn(input_context):
            del input_context
            x = tf.random.uniform((10, 10))
            y = tf.random.uniform((10,))
            return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat().batch(2)
    return dataset_fn