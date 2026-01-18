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
class TFCvtSelfAttentionConvProjection(keras.layers.Layer):
    """Convolutional projection layer."""

    def __init__(self, config: CvtConfig, embed_dim: int, kernel_size: int, stride: int, padding: int, **kwargs):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=padding)
        self.convolution = keras.layers.Conv2D(filters=embed_dim, kernel_size=kernel_size, kernel_initializer=get_initializer(config.initializer_range), padding='valid', strides=stride, use_bias=False, name='convolution', groups=embed_dim)
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name='normalization')
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_state = self.convolution(self.padding(hidden_state))
        hidden_state = self.normalization(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution', None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.embed_dim])
        if getattr(self, 'normalization', None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.embed_dim])