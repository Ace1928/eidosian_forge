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
class TFCvtSelfAttention(keras.layers.Layer):
    """
    Self-attention layer. A depth-wise separable convolution operation (Convolutional Projection), is applied for
    query, key, and value embeddings.
    """

    def __init__(self, config: CvtConfig, num_heads: int, embed_dim: int, kernel_size: int, stride_q: int, stride_kv: int, padding_q: int, padding_kv: int, qkv_projection_method: str, qkv_bias: bool, attention_drop_rate: float, with_cls_token: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.scale = embed_dim ** (-0.5)
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.convolution_projection_query = TFCvtSelfAttentionProjection(config, embed_dim, kernel_size, stride_q, padding_q, projection_method='linear' if qkv_projection_method == 'avg' else qkv_projection_method, name='convolution_projection_query')
        self.convolution_projection_key = TFCvtSelfAttentionProjection(config, embed_dim, kernel_size, stride_kv, padding_kv, projection_method=qkv_projection_method, name='convolution_projection_key')
        self.convolution_projection_value = TFCvtSelfAttentionProjection(config, embed_dim, kernel_size, stride_kv, padding_kv, projection_method=qkv_projection_method, name='convolution_projection_value')
        self.projection_query = keras.layers.Dense(units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), use_bias=qkv_bias, bias_initializer='zeros', name='projection_query')
        self.projection_key = keras.layers.Dense(units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), use_bias=qkv_bias, bias_initializer='zeros', name='projection_key')
        self.projection_value = keras.layers.Dense(units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), use_bias=qkv_bias, bias_initializer='zeros', name='projection_value')
        self.dropout = keras.layers.Dropout(attention_drop_rate)

    def rearrange_for_multi_head_attention(self, hidden_state: tf.Tensor) -> tf.Tensor:
        batch_size, hidden_size, _ = shape_list(hidden_state)
        head_dim = self.embed_dim // self.num_heads
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, self.num_heads, head_dim))
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 1, 3))
        return hidden_state

    def call(self, hidden_state: tf.Tensor, height: int, width: int, training: bool=False) -> tf.Tensor:
        if self.with_cls_token:
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)
        batch_size, hidden_size, num_channels = shape_list(hidden_state)
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))
        key = self.convolution_projection_key(hidden_state, training=training)
        query = self.convolution_projection_query(hidden_state, training=training)
        value = self.convolution_projection_value(hidden_state, training=training)
        if self.with_cls_token:
            query = tf.concat((cls_token, query), axis=1)
            key = tf.concat((cls_token, key), axis=1)
            value = tf.concat((cls_token, value), axis=1)
        head_dim = self.embed_dim // self.num_heads
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))
        attention_score = tf.matmul(query, key, transpose_b=True) * self.scale
        attention_probs = stable_softmax(logits=attention_score, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        context = tf.matmul(attention_probs, value)
        _, _, hidden_size, _ = shape_list(context)
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        context = tf.reshape(context, (batch_size, hidden_size, self.num_heads * head_dim))
        return context

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution_projection_query', None) is not None:
            with tf.name_scope(self.convolution_projection_query.name):
                self.convolution_projection_query.build(None)
        if getattr(self, 'convolution_projection_key', None) is not None:
            with tf.name_scope(self.convolution_projection_key.name):
                self.convolution_projection_key.build(None)
        if getattr(self, 'convolution_projection_value', None) is not None:
            with tf.name_scope(self.convolution_projection_value.name):
                self.convolution_projection_value.build(None)
        if getattr(self, 'projection_query', None) is not None:
            with tf.name_scope(self.projection_query.name):
                self.projection_query.build([None, None, self.embed_dim])
        if getattr(self, 'projection_key', None) is not None:
            with tf.name_scope(self.projection_key.name):
                self.projection_key.build([None, None, self.embed_dim])
        if getattr(self, 'projection_value', None) is not None:
            with tf.name_scope(self.projection_value.name):
                self.projection_value.build([None, None, self.embed_dim])