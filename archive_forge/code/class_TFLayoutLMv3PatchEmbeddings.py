from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
class TFLayoutLMv3PatchEmbeddings(keras.layers.Layer):
    """LayoutLMv3 image (patch) embeddings."""

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        patch_sizes = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)
        self.proj = keras.layers.Conv2D(filters=config.hidden_size, kernel_size=patch_sizes, strides=patch_sizes, padding='valid', data_format='channels_last', use_bias=True, kernel_initializer=get_initializer(config.initializer_range), name='proj')
        self.hidden_size = config.hidden_size
        self.num_patches = config.input_size ** 2 // (patch_sizes[0] * patch_sizes[1])
        self.config = config

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])
        embeddings = self.proj(pixel_values)
        embeddings = tf.reshape(embeddings, (-1, self.num_patches, self.hidden_size))
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'proj', None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, None, self.config.num_channels])