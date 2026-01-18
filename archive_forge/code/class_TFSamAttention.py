from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamAttention(keras.layers.Layer):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(self, config, downsample_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate
        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError('num_attention_heads must divide hidden_size.')
        self.q_proj = keras.layers.Dense(self.internal_dim, name='q_proj')
        self.k_proj = keras.layers.Dense(self.internal_dim, name='k_proj')
        self.v_proj = keras.layers.Dense(self.internal_dim, name='v_proj')
        self.out_proj = keras.layers.Dense(self.hidden_size, name='out_proj')

    def _separate_heads(self, hidden_states: tf.Tensor, num_attention_heads: int) -> tf.Tensor:
        batch, point_batch_size, n_tokens, channel = shape_list(hidden_states)
        c_per_head = channel // num_attention_heads
        hidden_states = tf.reshape(hidden_states, (batch * point_batch_size, n_tokens, num_attention_heads, c_per_head))
        return tf.transpose(hidden_states, perm=[0, 2, 1, 3])

    def _recombine_heads(self, hidden_states: tf.Tensor, point_batch_size: int) -> tf.Tensor:
        batch, n_heads, n_tokens, c_per_head = shape_list(hidden_states)
        hidden_states = tf.transpose(hidden_states, perm=[0, 2, 1, 3])
        return tf.reshape(hidden_states, (batch // tf.reduce_max([1, point_batch_size]), point_batch_size, n_tokens, n_heads * c_per_head))

    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        point_batch_size = shape_list(query)[1]
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)
        _, _, _, c_per_head = shape_list(query)
        attn = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))
        attn = attn / tf.math.sqrt(float(c_per_head))
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, value)
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)
        return out

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'q_proj', None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.hidden_size])
        if getattr(self, 'k_proj', None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.hidden_size])
        if getattr(self, 'v_proj', None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.hidden_size])
        if getattr(self, 'out_proj', None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.internal_dim])