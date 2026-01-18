from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_gptj import GPTJConfig
class TFGPTJMLP(keras.layers.Layer):

    def __init__(self, intermediate_size: int, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        embed_dim = config.n_embd
        self.fc_in = keras.layers.Dense(intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='fc_in')
        self.fc_out = keras.layers.Dense(embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='fc_out')
        self.act = get_tf_activation(config.activation_function)
        self.dropout = keras.layers.Dropout(config.embd_pdrop)
        self.embed_dim = config.n_embd
        self.intermediate_size = intermediate_size

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'fc_in', None) is not None:
            with tf.name_scope(self.fc_in.name):
                self.fc_in.build([None, None, self.embed_dim])
        if getattr(self, 'fc_out', None) is not None:
            with tf.name_scope(self.fc_out.name):
                self.fc_out.build([None, None, self.intermediate_size])