from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
class TFSwinPatchEmbeddings(keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        image_size, patch_size = (config.image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.embed_dim)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.projection = keras.layers.Conv2D(filters=hidden_size, kernel_size=self.patch_size, strides=self.patch_size, padding='valid', name='projection')

    def maybe_pad(self, pixel_values: tf.Tensor, height: int, width: int) -> tf.Tensor:
        if width % self.patch_size[1] != 0:
            pad_values = ((0, 0), (0, 0), (0, 0), (0, self.patch_size[1] - width % self.patch_size[1]))
            pixel_values = tf.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = ((0, 0), (0, 0), (0, self.patch_size[0] - height % self.patch_size[0]), (0, 0))
            pixel_values = tf.pad(pixel_values, pad_values)
        return pixel_values

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> Tuple[tf.Tensor, Tuple[int, int]]:
        _, num_channels, height, width = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        pixel_values = self.maybe_pad(pixel_values, height, width)
        pixel_values = tf.transpose(pixel_values, (0, 2, 3, 1))
        embeddings = self.projection(pixel_values, training=training)
        embeddings = tf.transpose(embeddings, (0, 3, 1, 2))
        batch_size, channels, height, width = shape_list(embeddings)
        output_dimensions = (height, width)
        embeddings = tf.reshape(embeddings, (batch_size, channels, -1))
        embeddings = tf.transpose(embeddings, (0, 2, 1))
        return (embeddings, output_dimensions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])