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
class TFSamVisionLayer(keras.layers.Layer):

    def __init__(self, config, window_size, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm1')
        self.attn = TFSamVisionAttention(config, window_size, name='attn')
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm2')
        self.mlp = TFSamMLPBlock(config, name='mlp')
        self.window_size = window_size
        self.config = config

    def window_partition(self, hidden_states: tf.Tensor, window_size: int) -> Tuple[tf.Tensor, Tuple[int, int]]:
        batch_size, height, width, channel = shape_list(hidden_states)
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            hidden_states = tf.pad(hidden_states, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        pad_height, pad_width = (height + pad_h, width + pad_w)
        hidden_states = tf.reshape(hidden_states, [batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel])
        windows = tf.reshape(tf.transpose(hidden_states, perm=[0, 1, 3, 2, 4, 5]), [-1, window_size, window_size, channel])
        return (windows, (pad_height, pad_width))

    def window_unpartition(self, windows: tf.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]) -> tf.Tensor:
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = shape_list(windows)[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = tf.reshape(windows, [batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1])
        hidden_states = tf.reshape(tf.transpose(hidden_states, perm=[0, 1, 3, 2, 4, 5]), [batch_size, pad_height, pad_width, -1])
        if pad_height > height or pad_width > width:
            hidden_states = hidden_states[:, :height, :width, :]
        return hidden_states

    def call(self, hidden_states: tf.Tensor, output_attentions: Optional[bool]=False, training: Optional[bool]=False) -> Tuple[tf.Tensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            height, width = (hidden_states.shape[1], hidden_states.shape[2])
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)
        hidden_states, attn_weights = self.attn(hidden_states=hidden_states, output_attentions=output_attentions, training=training)
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))
        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norm1', None) is not None:
            with tf.name_scope(self.layer_norm1.name):
                self.layer_norm1.build([None, None, None, self.config.hidden_size])
        if getattr(self, 'attn', None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        if getattr(self, 'layer_norm2', None) is not None:
            with tf.name_scope(self.layer_norm2.name):
                self.layer_norm2.build([None, None, None, self.config.hidden_size])
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)