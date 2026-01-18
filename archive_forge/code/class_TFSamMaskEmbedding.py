from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamMaskEmbedding(keras.layers.Layer):

    def __init__(self, config: SamPromptEncoderConfig, **kwargs):
        super().__init__(**kwargs)
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = keras.layers.Conv2D(self.mask_input_channels, kernel_size=2, strides=2, name='conv1')
        self.conv2 = keras.layers.Conv2D(config.mask_input_channels, kernel_size=2, strides=2, name='conv2')
        self.conv3 = keras.layers.Conv2D(config.hidden_size, kernel_size=1, name='conv3')
        self.layer_norm1 = TFSamLayerNorm(self.mask_input_channels, config.layer_norm_eps, name='layer_norm1')
        self.layer_norm2 = TFSamLayerNorm(self.mask_input_channels * 4, config.layer_norm_eps, name='layer_norm2')
        self.config = config

    def call(self, masks):
        masks = tf.transpose(masks, perm=(0, 2, 3, 1))
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        dense_embeddings = tf.transpose(dense_embeddings, perm=(0, 3, 1, 2))
        return dense_embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        with tf.name_scope('conv1'):
            self.conv1.build([None, None, None, 1])
        with tf.name_scope('conv2'):
            self.conv2.build([None, None, None, self.mask_input_channels])
        with tf.name_scope('conv3'):
            self.conv3.build([None, None, None, self.mask_input_channels * 4])
        with tf.name_scope('layer_norm1'):
            self.layer_norm1.build([None, None, None, self.mask_input_channels])
        with tf.name_scope('layer_norm2'):
            self.layer_norm2.build([None, None, None, self.mask_input_channels * 4])