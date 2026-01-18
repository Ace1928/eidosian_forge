from __future__ import annotations
import collections.abc
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
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class TFGroupViTVisionEmbeddings(keras.layers.Layer):
    """
    Construct the position and patch embeddings.

    """

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.patch_embeddings = TFGroupViTPatchEmbeddings(config, name='patch_embeddings')
        self.dropout = keras.layers.Dropout(rate=config.dropout, name='dropout')
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.config = config

    def build(self, input_shape=None):
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.add_weight(shape=(1, num_patches, self.config.hidden_size), initializer='zeros', trainable=True, name='position_embeddings')
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embeddings', None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.hidden_size])

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        batch_size, num_patches, dim = shape_list(embeddings)
        num_positions = shape_list(self.position_embeddings)[1]
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tf.image.resize(images=tf.reshape(patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)), size=(h0, w0), method='bicubic')
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return patch_pos_embed

    def call(self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool=False, training: bool=False) -> tf.Tensor:
        _, _, height, width = shape_list(pixel_values)
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        embeddings = self.layernorm(embeddings)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings