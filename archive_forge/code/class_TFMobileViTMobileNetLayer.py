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
class TFMobileViTMobileNetLayer(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int=1, num_stages: int=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = []
        for i in range(num_stages):
            layer = TFMobileViTInvertedResidual(config, in_channels=in_channels, out_channels=out_channels, stride=stride if i == 0 else 1, name=f'layer.{i}')
            self.layers.append(layer)
            in_channels = out_channels

    def call(self, features: tf.Tensor, training: bool=False) -> tf.Tensor:
        for layer_module in self.layers:
            features = layer_module(features, training=training)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layers', None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)