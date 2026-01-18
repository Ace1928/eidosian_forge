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
class TFSamVisionNeck(keras.layers.Layer):

    def __init__(self, config: SamVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.conv1 = keras.layers.Conv2D(config.output_channels, kernel_size=1, use_bias=False, name='conv1')
        self.layer_norm1 = TFSamLayerNorm(config.output_channels, name='layer_norm1')
        self.conv2 = keras.layers.Conv2D(config.output_channels, kernel_size=3, padding='same', use_bias=False, name='conv2')
        self.layer_norm2 = TFSamLayerNorm(config.output_channels, name='layer_norm2')

    def call(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = tf.transpose(hidden_states, perm=[0, 3, 1, 2])
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv1', None) is not None:
            with tf.name_scope(self.conv1.name):
                self.conv1.build([None, None, None, self.config.hidden_size])
        if getattr(self, 'layer_norm1', None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build(None)
        if getattr(self, 'conv2', None) is not None:
            with tf.name_scope(self.conv2.name):
                self.conv2.build([None, None, None, self.config.output_channels])
        if getattr(self, 'layer_norm2', None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build(None)