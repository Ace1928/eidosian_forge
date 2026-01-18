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
class TFTransformerBlock(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.dropout = keras.layers.Dropout(config.dropout)
        self.activation = config.activation
        self.output_attentions = config.output_attentions
        assert config.dim % config.n_heads == 0, f'Hidden size {config.dim} not dividable by number of heads {config.n_heads}'
        self.attention = TFMultiHeadSelfAttention(config, name='attention')
        self.sa_layer_norm = keras.layers.LayerNormalization(epsilon=1e-12, name='sa_layer_norm')
        self.ffn = TFFFN(config, name='ffn')
        self.output_layer_norm = keras.layers.LayerNormalization(epsilon=1e-12, name='output_layer_norm')
        self.config = config

    def call(self, x, attn_mask, head_mask, output_attentions, training=False):
        """
        Parameters:
            x: tf.Tensor(bs, seq_length, dim)
            attn_mask: tf.Tensor(bs, seq_length)

        Outputs: sa_weights: tf.Tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
        tf.Tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        sa_output = self.attention(x, x, x, attn_mask, head_mask, output_attentions, training=training)
        if output_attentions:
            sa_output, sa_weights = sa_output
        else:
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)
        ffn_output = self.ffn(sa_output, training=training)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'sa_layer_norm', None) is not None:
            with tf.name_scope(self.sa_layer_norm.name):
                self.sa_layer_norm.build([None, None, self.config.dim])
        if getattr(self, 'ffn', None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)
        if getattr(self, 'output_layer_norm', None) is not None:
            with tf.name_scope(self.output_layer_norm.name):
                self.output_layer_norm.build([None, None, self.config.dim])