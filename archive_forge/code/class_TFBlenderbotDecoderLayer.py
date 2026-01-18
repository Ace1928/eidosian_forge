from __future__ import annotations
import os
import random
import warnings
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_blenderbot import BlenderbotConfig
class TFBlenderbotDecoderLayer(keras.layers.Layer):

    def __init__(self, config: BlenderbotConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.d_model
        self.self_attn = TFBlenderbotAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, name='self_attn', is_decoder=True)
        self.dropout = keras.layers.Dropout(config.dropout)
        self.activation_fn = get_tf_activation(config.activation_function)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        self.self_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='self_attn_layer_norm')
        self.encoder_attn = TFBlenderbotAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, name='encoder_attn', is_decoder=True)
        self.encoder_attn_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='encoder_attn_layer_norm')
        self.fc1 = keras.layers.Dense(config.decoder_ffn_dim, name='fc1')
        self.fc2 = keras.layers.Dense(self.embed_dim, name='fc2')
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='final_layer_norm')
        self.config = config

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, encoder_hidden_states: tf.Tensor | None=None, encoder_attention_mask: tf.Tensor | None=None, layer_head_mask: tf.Tensor | None=None, cross_attn_layer_head_mask: tf.Tensor | None=None, past_key_value: Tuple[tf.Tensor] | None=None, training: Optional[bool]=False) -> Tuple[tf.Tensor, tf.Tensor, Tuple[Tuple[tf.Tensor]]]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`tf.Tensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            encoder_hidden_states (`tf.Tensor`):
                cross attention input to the layer of shape *(batch, seq_len, embed_dim)*
            encoder_attention_mask (`tf.Tensor`): encoder attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`tf.Tensor`): mask for attention heads in a given layer of size
                *(decoder_attention_heads,)*
            cross_attn_layer_head_mask (`tf.Tensor`): mask for heads of the cross-attention module.
                *(decoder_attention_heads,)*
            past_key_value (`Tuple(tf.Tensor)`): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask, layer_head_mask=layer_head_mask)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask, past_key_value=cross_attn_past_key_value)
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = residual + hidden_states
            present_key_value = present_key_value + cross_attn_present_key_value
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = residual + hidden_states
        return (hidden_states, self_attn_weights, cross_attn_weights, present_key_value)

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
        if getattr(self, 'encoder_attn', None) is not None:
            with tf.name_scope(self.encoder_attn.name):
                self.encoder_attn.build(None)
        if getattr(self, 'encoder_attn_layer_norm', None) is not None:
            with tf.name_scope(self.encoder_attn_layer_norm.name):
                self.encoder_attn_layer_norm.build([None, None, self.embed_dim])
        if getattr(self, 'fc1', None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.embed_dim])
        if getattr(self, 'fc2', None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.config.decoder_ffn_dim])
        if getattr(self, 'final_layer_norm', None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])