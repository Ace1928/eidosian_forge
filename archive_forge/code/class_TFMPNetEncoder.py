from __future__ import annotations
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mpnet import MPNetConfig
class TFMPNetEncoder(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.n_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.initializer_range = config.initializer_range
        self.layer = [TFMPNetLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        with tf.name_scope('relative_attention_bias'):
            self.relative_attention_bias = self.add_weight(name='embeddings', shape=[self.relative_attention_num_buckets, self.n_heads], initializer=get_initializer(self.initializer_range))
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=False):
        position_bias = self.compute_position_bias(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions, position_bias=position_bias, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += tf.cast(tf.math.less(n, 0), dtype=relative_position.dtype) * num_buckets
        n = tf.math.abs(n)
        max_exact = num_buckets // 2
        is_small = tf.math.less(n, max_exact)
        val_if_large = max_exact + tf.cast(tf.math.log(n / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact), dtype=relative_position.dtype)
        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        ret += tf.where(is_small, n, val_if_large)
        return ret

    def compute_position_bias(self, x, position_ids=None):
        """Compute binned relative position bias"""
        input_shape = shape_list(x)
        qlen, klen = (input_shape[1], input_shape[1])
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = tf.range(qlen)[:, None]
            memory_position = tf.range(klen)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position, num_buckets=self.relative_attention_num_buckets)
        values = tf.gather(self.relative_attention_bias, rp_bucket)
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
        return values