from __future__ import annotations
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlm import XLMConfig
class TFXLMMultiHeadAttention(keras.layers.Layer):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = next(TFXLMMultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.output_attentions = config.output_attentions
        assert self.dim % self.n_heads == 0
        self.q_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='q_lin')
        self.k_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='k_lin')
        self.v_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='v_lin')
        self.out_lin = keras.layers.Dense(dim, kernel_initializer=get_initializer(config.init_std), name='out_lin')
        self.dropout = keras.layers.Dropout(config.attention_dropout)
        self.pruned_heads = set()
        self.dim = dim

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input, mask, kv, cache, head_mask, output_attentions, training=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        bs, qlen, dim = shape_list(input)
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = shape_list(kv)[1]
        dim_per_head = self.dim // self.n_heads
        mask_reshape = (bs, 1, qlen, klen) if len(shape_list(mask)) == 3 else (bs, 1, 1, klen)

        def shape(x):
            """projection"""
            return tf.transpose(tf.reshape(x, (bs, -1, self.n_heads, dim_per_head)), perm=(0, 2, 1, 3))

        def unshape(x):
            """compute context"""
            return tf.reshape(tf.transpose(x, perm=(0, 2, 1, 3)), (bs, -1, self.n_heads * dim_per_head))
        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = tf.concat([k_, k], axis=2)
                    v = tf.concat([v_, v], axis=2)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)
        f_dim_per_head = tf.cast(dim_per_head, dtype=q.dtype)
        q = tf.multiply(q, tf.math.rsqrt(f_dim_per_head))
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
        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'q_lin', None) is not None:
            with tf.name_scope(self.q_lin.name):
                self.q_lin.build([None, None, self.dim])
        if getattr(self, 'k_lin', None) is not None:
            with tf.name_scope(self.k_lin.name):
                self.k_lin.build([None, None, self.dim])
        if getattr(self, 'v_lin', None) is not None:
            with tf.name_scope(self.v_lin.name):
                self.v_lin.build([None, None, self.dim])
        if getattr(self, 'out_lin', None) is not None:
            with tf.name_scope(self.out_lin.name):
                self.out_lin.build([None, None, self.dim])