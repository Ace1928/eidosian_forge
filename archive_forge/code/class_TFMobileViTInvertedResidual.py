from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTInvertedResidual(keras.layers.Layer):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int=1, **kwargs) -> None:
        super().__init__(**kwargs)
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)
        if stride not in [1, 2]:
            raise ValueError(f'Invalid stride {stride}.')
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand_1x1 = TFMobileViTConvLayer(config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1, name='expand_1x1')
        self.conv_3x3 = TFMobileViTConvLayer(config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=3, stride=stride, groups=expanded_channels, dilation=dilation, name='conv_3x3')
        self.reduce_1x1 = TFMobileViTConvLayer(config, in_channels=expanded_channels, out_channels=out_channels, kernel_size=1, use_activation=False, name='reduce_1x1')

    def call(self, features: tf.Tensor, training: bool=False) -> tf.Tensor:
        residual = features
        features = self.expand_1x1(features, training=training)
        features = self.conv_3x3(features, training=training)
        features = self.reduce_1x1(features, training=training)
        return residual + features if self.use_residual else features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'expand_1x1', None) is not None:
            with tf.name_scope(self.expand_1x1.name):
                self.expand_1x1.build(None)
        if getattr(self, 'conv_3x3', None) is not None:
            with tf.name_scope(self.conv_3x3.name):
                self.conv_3x3.build(None)
        if getattr(self, 'reduce_1x1', None) is not None:
            with tf.name_scope(self.reduce_1x1.name):
                self.reduce_1x1.build(None)