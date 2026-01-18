from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_convnext import ConvNextConfig
class TFConvNextEmbeddings(keras.layers.Layer):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config: ConvNextConfig, **kwargs):
        super().__init__(**kwargs)
        self.patch_embeddings = keras.layers.Conv2D(filters=config.hidden_sizes[0], kernel_size=config.patch_size, strides=config.patch_size, name='patch_embeddings', kernel_initializer=get_initializer(config.initializer_range), bias_initializer=keras.initializers.Zeros())
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-06, name='layernorm')
        self.num_channels = config.num_channels
        self.config = config

    def call(self, pixel_values):
        if isinstance(pixel_values, dict):
            pixel_values = pixel_values['pixel_values']
        tf.debugging.assert_equal(shape_list(pixel_values)[1], self.num_channels, message='Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'patch_embeddings', None) is not None:
            with tf.name_scope(self.patch_embeddings.name):
                self.patch_embeddings.build([None, None, None, self.config.num_channels])
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.config.hidden_sizes[0]])