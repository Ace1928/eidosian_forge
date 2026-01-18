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
class TFGPTJAttention(keras.layers.Layer):

    def __init__(self, config: GPTJConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads}).')
        self.scale_attn = self.head_dim ** 0.5
        self.rotary_dim = config.rotary_dim
        self.attn_dropout = keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = keras.layers.Dropout(config.resid_pdrop)
        self.q_proj = keras.layers.Dense(self.embed_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='q_proj')
        self.k_proj = keras.layers.Dense(self.embed_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='k_proj')
        self.v_proj = keras.layers.Dense(self.embed_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='v_proj')
        self.out_proj = keras.layers.Dense(self.embed_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='out_proj')
        self.max_positions = config.max_position_embeddings
        self.lower_triangle_mask = tf.reshape(tf.cast(tf.experimental.numpy.tril(tf.ones((self.max_positions, self.max_positions))), tf.int8), (1, 1, self.max_positions, self.max_positions))
        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(self.max_positions, pos_embd_dim)

    def get_causal_mask(self, key_length, query_length) -> tf.Tensor:
        return tf.cast(self.lower_triangle_mask[:, :, key_length - query_length:key_length, :key_length], tf.bool)

    @staticmethod
    def get_masked_bias(dtype: tf.DType) -> tf.Tensor:
        return tf.cast(tf.constant(-1000000000.0), dtype)

    def _split_heads(self, hidden_states: tf.Tensor, rotary: bool) -> tf.Tensor:
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = shape_list(hidden_states)[:-1] + [self.num_attention_heads, self.head_dim]
        hidden_states = tf.reshape(hidden_states, new_shape)
        if rotary:
            return hidden_states
        if len(shape_list(hidden_states)) == 4:
            return tf.transpose(hidden_states, (0, 2, 1, 3))
        if len(shape_list(hidden_states)) == 5:
            return tf.transpose(hidden_states, (0, 1, 3, 2, 4))
        raise ValueError(f'Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}')

    def _merge_heads(self, hidden_states: tf.Tensor) -> tf.Tensor:
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(shape_list(hidden_states)) == 4:
            hidden_states = tf.transpose(hidden_states, (0, 2, 1, 3))
        elif len(shape_list(hidden_states)) == 5:
            hidden_states = tf.transpose(hidden_states, (0, 1, 3, 2, 4))
        else:
            raise ValueError(f'Input tensor rank should be one of [4, 5], but is: {len(shape_list(hidden_states))}')
        new_shape = shape_list(hidden_states)[:-2] + [self.num_attention_heads * self.head_dim]
        return tf.reshape(hidden_states, new_shape)

    def _attn(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None) -> Tuple[tf.Tensor, tf.Tensor]:
        query_length, key_length = (shape_list(query)[-2], shape_list(key)[-2])
        causal_mask = self.get_causal_mask(key_length, query_length)
        query = tf.cast(query, tf.float32)
        key = tf.cast(key, tf.float32)
        attn_weights = tf.matmul(query, key, transpose_b=True)
        attn_weights = tf.where(causal_mask, attn_weights, self.get_masked_bias(attn_weights.dtype))
        attn_weights = attn_weights / self.scale_attn
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = stable_softmax(attn_weights, axis=-1)
        attn_weights = tf.cast(attn_weights, value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = tf.matmul(attn_weights, value)
        return (attn_output, attn_weights)

    def call(self, hidden_states: tf.Tensor, layer_past: Optional[Tuple[tf.Tensor, tf.Tensor]]=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, use_cache: bool=False, output_attentions: bool=False):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query, True)
        key = self._split_heads(key, True)
        value = self._split_heads(value, False)
        sincos = tf.cast(tf.gather(self.embed_positions, position_ids, axis=0), hidden_states.dtype)
        sincos = tf.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, :self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]
            q_rot = query[:, :, :, :self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]
            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)
            key = tf.concat((k_rot, k_pass), axis=-1)
            query = tf.concat((q_rot, q_pass), axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)
        key = tf.transpose(key, (0, 2, 1, 3))
        query = tf.transpose(query, (0, 2, 1, 3))
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = tf.concat((past_key, key), axis=-2)
            value = tf.concat((past_value, value), axis=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'q_proj', None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.embed_dim])
        if getattr(self, 'k_proj', None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.embed_dim])
        if getattr(self, 'v_proj', None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.embed_dim])
        if getattr(self, 'out_proj', None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.embed_dim])