from __future__ import annotations
import collections.abc
import math
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_vit import ViTConfig
class TFViTPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        image_size, patch_size = (config.image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.config = config
        self.projection = keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid', data_format='channels_last', use_bias=True, kernel_initializer=get_initializer(self.config.initializer_range), bias_initializer='zeros', name='projection')

    def call(self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool=False, training: bool=False) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if not interpolate_pos_encoding:
            if tf.executing_eagerly():
                if height != self.image_size[0] or width != self.image_size[1]:
                    raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        projection = self.projection(pixel_values)
        num_patches = width // self.patch_size[1] * (height // self.patch_size[0])
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, num_patches, -1))
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, None, self.num_channels])