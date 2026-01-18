from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
class TFConv1dSubsampler(keras.layers.Layer):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """

    def __init__(self, config: Speech2TextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_layers = config.num_conv_layers
        self.in_channels = config.input_feat_per_channel * config.input_channels
        self.mid_channels = config.conv_channels
        self.out_channels = config.d_model
        self.kernel_sizes = config.conv_kernel_sizes
        self.conv_layers = [keras.layers.Conv1D(filters=self.mid_channels if i < self.num_layers - 1 else self.out_channels * 2, kernel_size=k, strides=2, name=f'conv_layers.{i}') for i, k in enumerate(self.kernel_sizes)]

    def call(self, input_features: tf.Tensor) -> tf.Tensor:
        hidden_states = tf.cast(input_features, tf.float32)
        for i, conv in enumerate(self.conv_layers):
            pad_len = self.kernel_sizes[i] // 2
            hidden_shapes = shape_list(hidden_states)
            hidden_states = tf.concat((tf.zeros((hidden_shapes[0], pad_len, hidden_shapes[2])), hidden_states, tf.zeros((hidden_shapes[0], pad_len, hidden_shapes[2]))), axis=1)
            hidden_states = conv(hidden_states)
            hidden_states = glu(hidden_states, axis=2)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv_layers', None) is not None:
            for i, layer in enumerate(self.conv_layers):
                with tf.name_scope(layer.name):
                    layer.build([None, None, self.in_channels] if i == 0 else [None, None, self.mid_channels // 2])