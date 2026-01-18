from __future__ import annotations
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_vit_mae import ViTMAEConfig
class TFViTMAEEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)
        self.patch_embeddings = TFViTMAEPatchEmbeddings(config, name='patch_embeddings')
        self.num_patches = self.patch_embeddings.num_patches
        self.config = config

    def build(self, input_shape=None):
        self.cls_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=tf.random_normal_initializer(stddev=self.config.initializer_range), trainable=True, name='cls_token')
        self.position_embeddings = self.add_weight(shape=(1, self.num_patches + 1, self.config.hidden_size), initializer='zeros', trainable=False, name='position_embeddings')
        pos_embed = get_2d_sincos_pos_embed(self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches ** 0.5), add_cls_token=True)[None, ...]
        self.position_embeddings.assign(pos_embed)
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embeddings', None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)

    def random_masking(self, sequence: tf.Tensor, noise: tf.Tensor | None=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`tf.Tensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = shape_list(sequence)
        len_keep = int(seq_length * (1 - self.config.mask_ratio))
        if noise is None:
            noise = tf.random.uniform(shape=(batch_size, seq_length), minval=0.0, maxval=1.0)
        ids_shuffle = tf.argsort(noise, axis=1)
        ids_restore = tf.argsort(ids_shuffle, axis=1)
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = tf.gather(sequence, axis=1, batch_dims=1, indices=ids_keep)
        mask_keep = tf.zeros((batch_size, len_keep))
        mask_remove = tf.ones((batch_size, seq_length - len_keep))
        mask = tf.concat([mask_keep, mask_remove], axis=-1)
        mask = tf.gather(mask, axis=1, batch_dims=1, indices=ids_restore)
        return (sequence_unmasked, mask, ids_restore)

    def call(self, pixel_values: tf.Tensor, noise: tf.Tensor=None) -> tf.Tensor:
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = embeddings + self.position_embeddings[:, 1:, :]
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = tf.tile(cls_token, (shape_list(embeddings)[0], 1, 1))
        embeddings = tf.concat([cls_tokens, embeddings], axis=1)
        return (embeddings, mask, ids_restore)