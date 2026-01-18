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
from .configuration_mpnet import MPNetConfig
class TFMPNetSelfAttention(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads}')
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.q = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='q')
        self.k = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='k')
        self.v = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='v')
        self.o = keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='o')
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        batch_size = shape_list(hidden_states)[0]
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        q = self.transpose_for_scores(q, batch_size)
        k = self.transpose_for_scores(k, batch_size)
        v = self.transpose_for_scores(v, batch_size)
        attention_scores = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(shape_list(k)[-1], attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)
        if position_bias is not None:
            attention_scores += position_bias
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        c = tf.matmul(attention_probs, v)
        c = tf.transpose(c, perm=[0, 2, 1, 3])
        c = tf.reshape(c, (batch_size, -1, self.all_head_size))
        o = self.o(c)
        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'q', None) is not None:
            with tf.name_scope(self.q.name):
                self.q.build([None, None, self.config.hidden_size])
        if getattr(self, 'k', None) is not None:
            with tf.name_scope(self.k.name):
                self.k.build([None, None, self.config.hidden_size])
        if getattr(self, 'v', None) is not None:
            with tf.name_scope(self.v.name):
                self.v.build([None, None, self.config.hidden_size])
        if getattr(self, 'o', None) is not None:
            with tf.name_scope(self.o.name):
                self.o.build([None, None, self.config.hidden_size])