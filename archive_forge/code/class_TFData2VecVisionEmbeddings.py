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
class TFData2VecVisionEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.patch_embeddings = TFData2VecVisionPatchEmbeddings(config, name='patch_embeddings')
        self.num_patches = self.patch_embeddings.num_patches
        self.config = config
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=tf.random_normal_initializer(stddev=self.config.initializer_range), trainable=True, name='cls_token')
        if self.config.use_mask_token:
            self.mask_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=tf.random_normal_initializer(stddev=self.config.initializer_range), trainable=True, name='mask_token')
        else:
            self.mask_token = None
        if self.config.use_absolute_position_embeddings:
            self.position_embeddings = self.add_weight(shape=(1, self.num_patches + 1, self.config.hidden_size), initializer=tf.random_normal_initializer(stddev=self.config.initializer_range), trainable=True, name='position_embeddings')
        else:
            self.position_embeddings = None
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embeddings', None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)

    def call(self, pixel_values: tf.Tensor, bool_masked_pos: tf.Tensor | None=None) -> tf.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, projection_dim = shape_list(embeddings)
        cls_tokens = tf.tile(self.cls_token, (batch_size, 1, 1))
        if bool_masked_pos is not None:
            mask_tokens = tf.broadcast_to(self.mask_token, (batch_size, seq_len, projection_dim))
            w = bool_masked_pos[..., None]
            w = tf.cast(w, mask_tokens.dtype)
            embeddings = embeddings * (1 - w) + mask_tokens * w
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings