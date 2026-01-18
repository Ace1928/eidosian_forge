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
class TFMobileViTASPPPooling(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.global_pool = keras.layers.GlobalAveragePooling2D(keepdims=True, name='global_pool')
        self.conv_1x1 = TFMobileViTConvLayer(config, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, use_normalization=True, use_activation='relu', name='conv_1x1')

    def call(self, features: tf.Tensor, training: bool=False) -> tf.Tensor:
        spatial_size = shape_list(features)[1:-1]
        features = self.global_pool(features)
        features = self.conv_1x1(features, training=training)
        features = tf.image.resize(features, size=spatial_size, method='bilinear')
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'global_pool', None) is not None:
            with tf.name_scope(self.global_pool.name):
                self.global_pool.build([None, None, None, None])
        if getattr(self, 'conv_1x1', None) is not None:
            with tf.name_scope(self.conv_1x1.name):
                self.conv_1x1.build(None)