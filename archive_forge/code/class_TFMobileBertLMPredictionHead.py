from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFMobileBertLMPredictionHead(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.transform = TFMobileBertPredictionHeadTransform(config, name='transform')
        self.config = config

    def build(self, input_shape=None):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        self.dense = self.add_weight(shape=(self.config.hidden_size - self.config.embedding_size, self.config.vocab_size), initializer='zeros', trainable=True, name='dense/weight')
        self.decoder = self.add_weight(shape=(self.config.vocab_size, self.config.embedding_size), initializer='zeros', trainable=True, name='decoder/weight')
        if self.built:
            return
        self.built = True
        if getattr(self, 'transform', None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    def get_output_embeddings(self):
        return self

    def set_output_embeddings(self, value):
        self.decoder = value
        self.config.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {'bias': self.bias}

    def set_bias(self, value):
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = tf.matmul(hidden_states, tf.concat([tf.transpose(self.decoder), self.dense], axis=0))
        hidden_states = hidden_states + self.bias
        return hidden_states