import collections
import math
import string
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.layers import activation
from keras.src.layers import core
from keras.src.layers import regularization
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _masked_softmax(self, attention_scores, attention_mask=None):
    if attention_mask is not None:
        mask_expansion_axis = -len(self._attention_axes) * 2 - 1
        for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
            attention_mask = tf.expand_dims(attention_mask, axis=mask_expansion_axis)
    return self._softmax(attention_scores, attention_mask)