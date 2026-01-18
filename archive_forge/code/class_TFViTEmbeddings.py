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
class TFViTEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        self.patch_embeddings = TFViTPatchEmbeddings(config, name='patch_embeddings')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def build(self, input_shape=None):
        num_patches = self.patch_embeddings.num_patches
        self.cls_token = self.add_weight(shape=(1, 1, self.config.hidden_size), initializer=get_initializer(self.config.initializer_range), trainable=True, name='cls_token')
        self.position_embeddings = self.add_weight(shape=(1, num_patches + 1, self.config.hidden_size), initializer=get_initializer(self.config.initializer_range), trainable=True, name='position_embeddings')
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embeddings', None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build(None)

    def interpolate_pos_encoding(self, embeddings, height, width) -> tf.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        batch_size, seq_len, dim = shape_list(embeddings)
        num_patches = seq_len - 1
        _, num_positions, _ = shape_list(self.position_embeddings)
        num_positions -= 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tf.image.resize(images=tf.reshape(patch_pos_embed, shape=(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)), size=(h0, w0), method='bicubic')
        shape = shape_list(patch_pos_embed)
        assert h0 == shape[-3] and w0 == shape[-2]
        patch_pos_embed = tf.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return tf.concat(values=(class_pos_embed, patch_pos_embed), axis=1)

    def call(self, pixel_values: tf.Tensor, interpolate_pos_encoding: bool=False, training: bool=False) -> tf.Tensor:
        batch_size, num_channels, height, width = shape_list(pixel_values)
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, training=training)
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)
        return embeddings