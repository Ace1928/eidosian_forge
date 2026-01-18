from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
class TFSwinSelfAttention(keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, num_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(f'The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})')
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        window_size = config.window_size
        self.window_size = window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        self.query = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), use_bias=config.qkv_bias, name='query')
        self.key = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), use_bias=config.qkv_bias, name='key')
        self.value = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), use_bias=config.qkv_bias, name='value')
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.relative_position_bias_table = self.add_weight(shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_attention_heads), initializer='zeros', name='relative_position_bias_table')
        self.relative_position_index = self.add_weight(shape=(self.window_size[0] ** 2, self.window_size[1] ** 2), trainable=False, dtype=tf.int32, name='relative_position_index')
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = tf.reshape(coords, (shape_list(coords)[0], -1))
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, (1, 2, 0))
        stack_0, stack_1 = tf.unstack(relative_coords, axis=2)
        stack_0 += self.window_size[0] - 1
        stack_0 *= 2 * self.window_size[1] - 1
        stack_1 += self.window_size[1] - 1
        relative_coords = tf.stack([stack_0, stack_1], axis=2)
        self.relative_position_index.assign(tf.cast(tf.reduce_sum(relative_coords, axis=-1), tf.int32))
        if self.built:
            return
        self.built = True
        if getattr(self, 'query', None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.all_head_size])
        if getattr(self, 'key', None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.all_head_size])
        if getattr(self, 'value', None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.all_head_size])

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        new_x_shape = shape_list(x)[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor, ...]:
        batch_size, dim, _ = shape_list(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, (0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, (-1,)))
        relative_position_bias = tf.reshape(relative_position_bias, (self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attention_scores = attention_scores + tf.expand_dims(relative_position_bias, 0)
        if attention_mask is not None:
            mask_shape = shape_list(attention_mask)[0]
            attention_scores = tf.reshape(attention_scores, (batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim))
            attention_mask = tf.expand_dims(attention_mask, 1)
            attention_mask = tf.expand_dims(attention_mask, 0)
            attention_scores = attention_scores + attention_mask
            attention_scores = tf.reshape(attention_scores, (-1, self.num_attention_heads, dim, dim))
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.all_head_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs