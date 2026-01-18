from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_led import LEDConfig
class TFLEDEncoderLayer(keras.layers.Layer):

    def __init__(self, config: LEDConfig, layer_id: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFLEDEncoderAttention(config, layer_id, name='self_attn')
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='self_attn_layer_norm')
        self.dropout = keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        self.fc1 = keras.layers.Dense(config.encoder_ffn_dim, name='fc1')
        self.fc2 = keras.layers.Dense(self.embed_dim, name='fc2')
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='final_layer_norm')
        self.config = config

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, layer_head_mask: tf.Tensor, is_index_masked: tf.Tensor, is_index_global_attn: tf.Tensor, is_global_attn: bool, training=False):
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                *(config.encoder_attention_heads,)*.
        """
        residual = hidden_states
        layer_outputs = self.self_attn([hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn], training=training)
        hidden_states = layer_outputs[0]
        tf.debugging.assert_equal(shape_list(hidden_states), shape_list(residual), message=f'Self attn modified the shape of query {shape_list(residual)} to {shape_list(hidden_states)}')
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states,) + layer_outputs[1:]

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self_attn', None) is not None:
            with tf.name_scope(self.self_attn.name):
                self.self_attn.build(None)
        if getattr(self, 'self_attn_layer_norm', None) is not None:
            with tf.name_scope(self.self_attn_layer_norm.name):
                self.self_attn_layer_norm.build([None, None, self.embed_dim])
        if getattr(self, 'fc1', None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        if getattr(self, 'fc2', None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.encoder_ffn_dim])
        if getattr(self, 'final_layer_norm', None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])