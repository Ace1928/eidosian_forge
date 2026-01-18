from __future__ import annotations
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig
class TFRobertaPreLayerNormLMHead(keras.layers.Layer):
    """RobertaPreLayerNorm Head for masked language modeling."""

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.dense = keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.act = get_tf_activation('gelu')
        self.decoder = input_embeddings

    def build(self, input_shape=None):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {'bias': self.bias}

    def set_bias(self, value):
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
        return hidden_states