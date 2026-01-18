from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTSelfAttention(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size {(hidden_size,)} is not a multiple of the number of attention heads {config.num_attention_heads}.')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        scale = tf.cast(self.attention_head_size, dtype=tf.float32)
        self.scale = tf.math.sqrt(scale)
        self.query = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name='query')
        self.key = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name='key')
        self.value = keras.layers.Dense(self.all_head_size, use_bias=config.qkv_bias, name='value')
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.hidden_size = hidden_size

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        batch_size = tf.shape(hidden_states)[0]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.scale
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, shape=(batch_size, -1, self.all_head_size))
        return context_layer

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'query', None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.hidden_size])
        if getattr(self, 'key', None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.hidden_size])
        if getattr(self, 'value', None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.hidden_size])