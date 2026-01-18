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
class TFCvtEmbeddings(keras.layers.Layer):
    """Construct the Convolutional Token Embeddings."""

    def __init__(self, config: CvtConfig, patch_size: int, num_channels: int, embed_dim: int, stride: int, padding: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.convolution_embeddings = TFCvtConvEmbeddings(config, patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, stride=stride, padding=padding, name='convolution_embeddings')
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_state = self.convolution_embeddings(pixel_values)
        hidden_state = self.dropout(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution_embeddings', None) is not None:
            with tf.name_scope(self.convolution_embeddings.name):
                self.convolution_embeddings.build(None)