from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
class TFCvtLayer(keras.layers.Layer):
    """
    Convolutional Transformer Block composed by attention layers, normalization and multi-layer perceptrons (mlps). It
    consists of 3 chunks : an attention layer, an intermediate dense layer and an output layer. This corresponds to the
    `Block` class in the original implementation.
    """

    def __init__(self, config: CvtConfig, num_heads: int, embed_dim: int, kernel_size: int, stride_q: int, stride_kv: int, padding_q: int, padding_kv: int, qkv_projection_method: str, qkv_bias: bool, attention_drop_rate: float, drop_rate: float, mlp_ratio: float, drop_path_rate: float, with_cls_token: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFCvtAttention(config, num_heads, embed_dim, kernel_size, stride_q, stride_kv, padding_q, padding_kv, qkv_projection_method, qkv_bias, attention_drop_rate, drop_rate, with_cls_token, name='attention')
        self.intermediate = TFCvtIntermediate(config, embed_dim, mlp_ratio, name='intermediate')
        self.dense_output = TFCvtOutput(config, embed_dim, mlp_ratio, drop_rate, name='output')
        self.drop_path = TFCvtDropPath(drop_path_rate, name='drop_path') if drop_path_rate > 0.0 else keras.layers.Activation('linear', name='drop_path')
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=1e-05, name='layernorm_before')
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=1e-05, name='layernorm_after')
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool=False) -> tf.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_state), height, width, training=training)
        attention_output = self.drop_path(attention_output, training=training)
        hidden_state = attention_output + hidden_state
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)
        layer_output = self.dense_output(layer_output, hidden_state)
        layer_output = self.drop_path(layer_output, training=training)
        return layer_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'intermediate', None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, 'dense_output', None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
        if getattr(self, 'drop_path', None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
        if getattr(self, 'layernorm_before', None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.embed_dim])
        if getattr(self, 'layernorm_after', None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.embed_dim])