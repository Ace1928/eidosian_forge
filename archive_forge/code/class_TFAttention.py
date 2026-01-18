from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_openai import OpenAIGPTConfig
class TFAttention(keras.layers.Layer):

    def __init__(self, nx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        n_state = nx
        assert n_state % config.n_head == 0, f'Hidden dimension {n_state} not dividable by number of heads {config.n_head}'
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions
        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name='c_attn')
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name='c_proj')
        self.attn_dropout = keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = keras.layers.Dropout(config.resid_pdrop)
        self.n_state = n_state
        self.pruned_heads = set()

    def prune_heads(self, heads):
        pass

    @staticmethod
    def causal_attention_mask(nd, ns):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return m

    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)
            w = w / tf.math.sqrt(dk)
        _, _, nd, ns = shape_list(w)
        b = tf.cast(self.causal_attention_mask(nd, ns), dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 10000.0 * (1 - b)
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask
        w = stable_softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)
        if head_mask is not None:
            w = w * head_mask
        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, x, attention_mask, head_mask, output_attentions, training=False):
        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)
        outputs = [a] + attn_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'c_attn', None) is not None:
            with tf.name_scope(self.c_attn.name):
                self.c_attn.build([None, None, self.n_state * 3])
        if getattr(self, 'c_proj', None) is not None:
            with tf.name_scope(self.c_proj.name):
                self.c_proj.build([None, None, self.n_state])