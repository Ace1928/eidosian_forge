from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_convnextv2 import ConvNextV2Config
class TFConvNextV2Stage(keras.layers.Layer):
    """ConvNextV2 stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config (`ConvNextV2V2Config`):
            Model configuration class.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        depth (`int`):
            Number of residual blocks.
        drop_path_rates(`List[float]`):
            Stochastic depth rates for each layer.
    """

    def __init__(self, config: ConvNextV2Config, in_channels: int, out_channels: int, kernel_size: int=2, stride: int=2, depth: int=2, drop_path_rates: Optional[List[float]]=None, **kwargs):
        super().__init__(**kwargs)
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = [keras.layers.LayerNormalization(epsilon=1e-06, name='downsampling_layer.0'), keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, kernel_initializer=get_initializer(config.initializer_range), bias_initializer=keras.initializers.Zeros(), name='downsampling_layer.1')]
        else:
            self.downsampling_layer = [tf.identity]
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = [TFConvNextV2Layer(config, dim=out_channels, drop_path=drop_path_rates[j], name=f'layers.{j}') for j in range(depth)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def call(self, hidden_states):
        for layer in self.downsampling_layer:
            hidden_states = layer(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if self.in_channels != self.out_channels or self.stride > 1:
            with tf.name_scope(self.downsampling_layer[0].name):
                self.downsampling_layer[0].build([None, None, None, self.in_channels])
            with tf.name_scope(self.downsampling_layer[1].name):
                self.downsampling_layer[1].build([None, None, None, self.in_channels])