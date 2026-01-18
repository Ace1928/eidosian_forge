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
def _find_repeated_tokens(self, vocabulary):
    """Return all repeated tokens in a vocabulary."""
    vocabulary_set = set(vocabulary)
    if len(vocabulary) != len(vocabulary_set):
        return [item for item, count in collections.Counter(vocabulary).items() if count > 1]
    else:
        return []