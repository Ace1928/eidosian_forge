from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class TFData2VecVisionPatchEmbeddings(keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        image_size, patch_size = (config.image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        self.num_channels = num_channels
        self.projection = keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid', data_format='channels_last', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='projection')

    def call(self, pixel_values: tf.Tensor, training: bool=False) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if tf.executing_eagerly():
            if num_channels != self.num_channels:
                raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        projection = self.projection(pixel_values)
        num_patches = width // self.patch_size[1] * (height // self.patch_size[0])
        return tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])