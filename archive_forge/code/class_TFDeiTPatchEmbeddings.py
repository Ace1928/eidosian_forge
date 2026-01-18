from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_deit import DeiTConfig
class TFDeiTPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        image_size, patch_size = (config.image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = keras.layers.Conv2D(hidden_size, kernel_size=patch_size, strides=patch_size, name='projection')

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        batch_size, height, width, num_channels = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if tf.executing_eagerly() and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        x = self.projection(pixel_values)
        batch_size, height, width, num_channels = shape_list(x)
        x = tf.reshape(x, (batch_size, height * width, num_channels))
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])