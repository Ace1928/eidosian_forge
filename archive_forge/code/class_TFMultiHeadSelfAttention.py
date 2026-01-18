from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_distilbert import DistilBertConfig
class TFMultiHeadSelfAttention(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = keras.layers.Dropout(config.attention_dropout)
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0, f'Hidden size {self.dim} not dividable by number of heads {self.n_heads}'
        self.q_lin = keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='q_lin')
        self.k_lin = keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='k_lin')
        self.v_lin = keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='v_lin')
        self.out_lin = keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='out_lin')
        self.pruned_heads = set()
        self.config = config

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, query, key, value, mask, head_mask, output_attentions, training=False):
        """
        Parameters:
            query: tf.Tensor(bs, seq_length, dim)
            key: tf.Tensor(bs, seq_length, dim)
            value: tf.Tensor(bs, seq_length, dim)
            mask: tf.Tensor(bs, seq_length)

        Returns:
            weights: tf.Tensor(bs, n_heads, seq_length, seq_length) Attention weights context: tf.Tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = shape_list(query)
        k_length = shape_list(key)[1]
        dim_per_head = int(self.dim / self.n_heads)
        dim_per_head = tf.cast(dim_per_head, dtype=tf.int32)
        mask_reshape = [bs, 1, 1, k_length]

        def shape(x):
            """separate heads"""
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            """group heads"""
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = tf.cast(q, dtype=tf.float32)
        q = tf.multiply(q, tf.math.rsqrt(tf.cast(dim_per_head, dtype=tf.float32)))
        k = tf.cast(k, dtype=q.dtype)
        scores = tf.matmul(q, k, transpose_b=True)
        mask = tf.reshape(mask, mask_reshape)
        mask = tf.cast(mask, dtype=scores.dtype)
        scores = scores - 1e+30 * (1.0 - mask)
        weights = stable_softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        if head_mask is not None:
            weights = weights * head_mask
        context = tf.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)
        if output_attentions:
            return (context, weights)
        else:
            return (context,)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'q_lin', None) is not None:
            with tf.name_scope(self.q_lin.name):
                self.q_lin.build([None, None, self.config.dim])
        if getattr(self, 'k_lin', None) is not None:
            with tf.name_scope(self.k_lin.name):
                self.k_lin.build([None, None, self.config.dim])
        if getattr(self, 'v_lin', None) is not None:
            with tf.name_scope(self.v_lin.name):
                self.v_lin.build([None, None, self.config.dim])
        if getattr(self, 'out_lin', None) is not None:
            with tf.name_scope(self.out_lin.name):
                self.out_lin.build([None, None, self.config.dim])