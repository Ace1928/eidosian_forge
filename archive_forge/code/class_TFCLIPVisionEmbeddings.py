from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
class TFCLIPVisionEmbeddings(keras.layers.Layer):

    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.config = config
        self.patch_embedding = keras.layers.Conv2D(filters=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, padding='valid', data_format='channels_last', use_bias=False, kernel_initializer=get_initializer(self.config.initializer_range * self.config.initializer_factor), name='patch_embedding')

    def build(self, input_shape: tf.TensorShape=None):
        factor = self.config.initializer_factor
        self.class_embedding = self.add_weight(shape=(self.embed_dim,), initializer=get_initializer(self.embed_dim ** (-0.5) * factor), trainable=True, name='class_embedding')
        with tf.name_scope('position_embedding'):
            self.position_embedding = self.add_weight(shape=(self.num_positions, self.embed_dim), initializer=get_initializer(self.config.initializer_range * factor), trainable=True, name='embeddings')
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embedding', None) is not None:
            with tf.name_scope(self.patch_embedding.name):
                self.patch_embedding.build([None, None, None, self.config.num_channels])

    def call(self, pixel_values: tf.Tensor) -> tf.Tensor:
        """`pixel_values` is expected to be of NCHW format."""
        batch_size, num_channels, height, width = shape_list(pixel_values)
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = tf.reshape(tensor=patch_embeds, shape=(batch_size, self.num_patches, -1))
        class_embeds = tf.broadcast_to(self.class_embedding, shape=(batch_size, 1, self.embed_dim))
        embeddings = tf.concat((class_embeds, patch_embeds), axis=1)
        embeddings = embeddings + self.position_embedding
        return embeddings