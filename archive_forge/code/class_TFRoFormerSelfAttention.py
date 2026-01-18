from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_roformer import RoFormerConfig
class TFRoFormerSelfAttention(keras.layers.Layer):

    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.query = keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        self.rotary_value = config.rotary_value
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, sinusoidal_pos: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        if sinusoidal_pos is not None:
            if self.rotary_value:
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer)
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)
        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(inputs=attention_probs, training=training)
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        sin, cos = tf.split(sinusoidal_pos, num_or_size_splits=2, axis=-1)
        sin_pos = tf.repeat(sin, 2, axis=-1)
        cos_pos = tf.repeat(cos, 2, axis=-1)
        rotate_half_query_layer = tf.stack([-query_layer[..., 1::2], query_layer[..., ::2]], axis=-1)
        rotate_half_query_layer = tf.reshape(rotate_half_query_layer, shape_list(query_layer))
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        rotate_half_key_layer = tf.stack([-key_layer[..., 1::2], key_layer[..., ::2]], axis=-1)
        rotate_half_key_layer = tf.reshape(rotate_half_key_layer, shape_list(key_layer))
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            rotate_half_value_layer = tf.stack([-value_layer[..., 1::2], value_layer[..., ::2]], axis=-1)
            rotate_half_value_layer = tf.reshape(rotate_half_value_layer, shape_list(value_layer))
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return (query_layer, key_layer, value_layer)
        return (query_layer, key_layer)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'query', None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, 'key', None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, 'value', None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])