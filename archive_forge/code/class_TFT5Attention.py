from __future__ import annotations
import copy
import itertools
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_t5 import T5Config
class TFT5Attention(keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFT5Attention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.has_relative_attention_bias = has_relative_attention_bias
        self.output_attentions = config.output_attentions
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        q_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor * (self.inner_dim * self.key_value_proj_dim) ** (-0.5))
        k_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor * self.inner_dim ** (-0.5))
        v_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor * self.inner_dim ** (-0.5))
        o_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor * self.inner_dim ** (-0.5))
        self.relative_attention_bias_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor * self.inner_dim ** (-0.5))
        self.q = keras.layers.Dense(self.inner_dim, use_bias=False, name='q', kernel_initializer=q_initializer)
        self.k = keras.layers.Dense(self.inner_dim, use_bias=False, name='k', kernel_initializer=k_initializer)
        self.v = keras.layers.Dense(self.inner_dim, use_bias=False, name='v', kernel_initializer=v_initializer)
        self.o = keras.layers.Dense(self.d_model, use_bias=False, name='o', kernel_initializer=o_initializer)
        self.dropout = keras.layers.Dropout(config.dropout_rate)
        self.pruned_heads = set()

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if self.has_relative_attention_bias:
            with tf.name_scope('relative_attention_bias'):
                self.relative_attention_bias = self.add_weight(name='embeddings', shape=[self.relative_attention_num_buckets, self.n_heads], initializer=self.relative_attention_bias_initializer)
        if getattr(self, 'q', None) is not None:
            with tf.name_scope(self.q.name):
                self.q.build([None, None, self.d_model])
        if getattr(self, 'k', None) is not None:
            with tf.name_scope(self.k.name):
                self.k.build([None, None, self.d_model])
        if getattr(self, 'v', None) is not None:
            with tf.name_scope(self.v.name):
                self.v.build([None, None, self.d_model])
        if getattr(self, 'o', None) is not None:
            with tf.name_scope(self.o.name):
                self.o.build([None, None, self.inner_dim])

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += tf.cast(tf.math.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(tf.math.log(tf.cast(relative_position, tf.float32) / tf.cast(max_exact, tf.float32)) / math.log(max_distance / max_exact) * (num_buckets - max_exact), dtype=relative_position.dtype)
        relative_position_if_large = tf.math.minimum(relative_position_if_large, num_buckets - 1)
        relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = tf.range(query_length)[:, None]
        memory_position = tf.range(key_length)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=not self.is_decoder, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        values = tf.gather(self.relative_attention_bias, relative_position_bucket)
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
        return values

    def call(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, training=False, output_attentions=False):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        batch_size, seq_length = shape_list(hidden_states)[:2]
        real_seq_length = seq_length
        if past_key_value is not None:
            assert len(past_key_value) == 2, f'past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states'
            real_seq_length += shape_list(past_key_value[0])[2] if query_length is None else query_length
        key_length = real_seq_length if key_value_states is None else shape_list(key_value_states)[1]

        def shape(hidden_states):
            """projection"""
            return tf.transpose(tf.reshape(hidden_states, (batch_size, -1, self.n_heads, self.key_value_proj_dim)), perm=(0, 2, 1, 3))

        def unshape(hidden_states):
            """compute context"""
            return tf.reshape(tf.transpose(hidden_states, perm=(0, 2, 1, 3)), (batch_size, -1, self.inner_dim))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                hidden_states = shape(proj_layer(key_value_states))
            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = tf.concat([past_key_value, hidden_states], axis=2)
                else:
                    hidden_states = past_key_value
            return hidden_states
        query_states = shape(self.q(hidden_states))
        key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)
        if self.is_decoder and use_cache:
            present_key_value_state = (key_states, value_states)
        else:
            present_key_value_state = None
        scores = tf.einsum('bnqd,bnkd->bnqk', query_states, key_states)
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = tf.zeros((1, self.n_heads, real_seq_length, key_length))
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)
            if past_key_value is not None:
                if not self.has_relative_attention_bias:
                    position_bias = position_bias[:, :, -seq_length:, :]
                else:
                    most_recently_filled_past_index = tf.reduce_max(tf.where(past_key_value[0][0, 0, :, 0] != 0.0))
                    position_bias = dynamic_slice(position_bias, (0, 0, most_recently_filled_past_index + 1, 0), (1, self.n_heads, seq_length, real_seq_length))
            if mask is not None:
                position_bias = tf.cast(position_bias, dtype=mask.dtype)
                position_bias = position_bias + mask
        scores += position_bias
        weights = stable_softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        if layer_head_mask is not None:
            tf.debugging.assert_equal(shape_list(layer_head_mask), [self.n_heads], message=f'Head mask for a single layer should be of size {self.n_heads}, but is {shape_list(layer_head_mask)}')
            weights = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * weights
        attn_output = tf.matmul(weights, value_states)
        attn_output = self.o(unshape(attn_output))
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs