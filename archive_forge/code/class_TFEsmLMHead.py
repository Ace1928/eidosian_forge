from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
class TFEsmLMHead(keras.layers.Layer):
    """ESM Head for masked language modeling."""

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.dense = keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        if config.tie_word_embeddings:
            self.decoder = None
        else:
            self.decoder = keras.layers.Dense(config.vocab_size, kernel_initializer=get_initializer(config.initializer_range), name='decoder', use_bias=False)
        self.config = config

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        self.bias = self.add_weight('bias', shape=(self.config.vocab_size,), initializer='zeros', trainable=True)
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        if getattr(self, 'decoder', None) is not None and (not self.config.tie_word_embeddings):
            with tf.name_scope(self.decoder.name):
                self.decoder.build([None, None, self.config.hidden_size])

    def get_bias(self):
        return {'bias': self.bias}

    def call(self, features):
        x = self.dense(features)
        x = tf.nn.gelu(x)
        x = self.layer_norm(x)
        if self.config.tie_word_embeddings:
            x = tf.matmul(x, self.decoder, transpose_b=True) + self.bias
        else:
            x = self.decoder(x) + self.bias
        return x