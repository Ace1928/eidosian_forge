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
class TFCvtConvEmbeddings(keras.layers.Layer):
    """Image to Convolution Embeddings. This convolutional operation aims to model local spatial contexts."""

    def __init__(self, config: CvtConfig, patch_size: int, num_channels: int, embed_dim: int, stride: int, padding: int, **kwargs):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=padding)
        self.patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.projection = keras.layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=stride, padding='valid', data_format='channels_last', kernel_initializer=get_initializer(config.initializer_range), name='projection')
        self.normalization = keras.layers.LayerNormalization(epsilon=1e-05, name='normalization')
        self.num_channels = num_channels
        self.embed_dim = embed_dim

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values['pixel_values']
        pixel_values = self.projection(self.padding(pixel_values))
        batch_size, height, width, num_channels = shape_list(pixel_values)
        hidden_size = height * width
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, hidden_size, num_channels))
        pixel_values = self.normalization(pixel_values)
        pixel_values = tf.reshape(pixel_values, shape=(batch_size, height, width, num_channels))
        return pixel_values

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])
        if getattr(self, 'normalization', None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, self.embed_dim])