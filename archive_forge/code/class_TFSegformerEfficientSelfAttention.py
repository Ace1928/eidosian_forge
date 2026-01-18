from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
class TFSegformerEfficientSelfAttention(keras.layers.Layer):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(self, config: SegformerConfig, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads ({self.num_attention_heads})')
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)
        self.query = keras.layers.Dense(self.all_head_size, name='query')
        self.key = keras.layers.Dense(self.all_head_size, name='key')
        self.value = keras.layers.Dense(self.all_head_size, name='value')
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = keras.layers.Conv2D(filters=hidden_size, kernel_size=sequence_reduction_ratio, strides=sequence_reduction_ratio, name='sr')
            self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm')

    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        batch_size = shape_list(tensor)[0]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool=False, training: bool=False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        batch_size = shape_list(hidden_states)[0]
        num_channels = shape_list(hidden_states)[2]
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        if self.sr_ratio > 1:
            hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            hidden_states = self.sr(hidden_states)
            hidden_states = tf.reshape(hidden_states, (batch_size, -1, num_channels))
            hidden_states = self.layer_norm(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        scale = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, scale)
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

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
        if getattr(self, 'sr', None) is not None:
            with tf.name_scope(self.sr.name):
                self.sr.build([None, None, None, self.hidden_size])
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.hidden_size])