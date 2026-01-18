import collections
import math
import os
import re
import unicodedata
from typing import List
import numpy as np
import six
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import constants
from autokeras.utils import data_utils
@keras.utils.register_keras_serializable()
class OnDeviceEmbedding(layers.Layer):
    """official.nlp.modeling.layers.OnDeviceEmbedding"""

    def __init__(self, vocab_size, embedding_width, initializer='glorot_uniform', use_one_hot=False, **kwargs):
        super(OnDeviceEmbedding, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding_width = embedding_width
        self._initializer = initializer
        self._use_one_hot = use_one_hot

    def get_config(self):
        config = {'vocab_size': self._vocab_size, 'embedding_width': self._embedding_width, 'initializer': self._initializer, 'use_one_hot': self._use_one_hot}
        base_config = super(OnDeviceEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.embeddings = self.add_weight('embeddings', shape=[self._vocab_size, self._embedding_width], initializer=self._initializer, dtype=tf.float32)
        super(OnDeviceEmbedding, self).build(input_shape)

    def call(self, inputs):
        flat_inputs = tf.reshape(inputs, [-1])
        if self._use_one_hot:
            one_hot_data = tf.one_hot(flat_inputs, depth=self._vocab_size, dtype=self.embeddings.dtype)
            embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            embeddings = tf.gather(self.embeddings, flat_inputs)
        embeddings = tf.reshape(embeddings, tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0))
        embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
        return embeddings