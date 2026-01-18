from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class TFData2VecVisionSelfAttention(keras.layers.Layer):

    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple]=None, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.query = keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key', use_bias=False)
        self.value = keras.layers.Dense(units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = keras.layers.Dropout(rate=config.attention_probs_dropout_prob)
        if window_size:
            self.relative_position_bias = TFData2VecVisionRelativePositionBias(config, window_size=window_size, name='relative_position_bias')
        else:
            self.relative_position_bias = None
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, relative_position_bias: Optional['TFData2VecVisionRelativePositionBias']=None, training: bool=False) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / self.sqrt_att_head_size
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias(0.0)[None, ...]
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(inputs=attention_probs, training=training)
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

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
        if getattr(self, 'relative_position_bias', None) is not None:
            with tf.name_scope(self.relative_position_bias.name):
                self.relative_position_bias.build(None)