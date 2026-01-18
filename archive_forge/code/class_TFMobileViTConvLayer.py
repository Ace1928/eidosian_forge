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
class TFMobileViTConvLayer(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, groups: int=1, bias: bool=False, dilation: int=1, use_normalization: bool=True, use_activation: Union[bool, str]=True, **kwargs) -> None:
        super().__init__(**kwargs)
        logger.warning(f'\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish to train/fine-tune this model, you need a GPU or a TPU')
        padding = int((kernel_size - 1) / 2) * dilation
        self.padding = keras.layers.ZeroPadding2D(padding)
        if out_channels % groups != 0:
            raise ValueError(f'Output channels ({out_channels}) are not divisible by {groups} groups.')
        self.convolution = keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding='VALID', dilation_rate=dilation, groups=groups, use_bias=bias, name='convolution')
        if use_normalization:
            self.normalization = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, name='normalization')
        else:
            self.normalization = None
        if use_activation:
            if isinstance(use_activation, str):
                self.activation = get_tf_activation(use_activation)
            elif isinstance(config.hidden_act, str):
                self.activation = get_tf_activation(config.hidden_act)
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, features: tf.Tensor, training: bool=False) -> tf.Tensor:
        padded_features = self.padding(features)
        features = self.convolution(padded_features)
        if self.normalization is not None:
            features = self.normalization(features, training=training)
        if self.activation is not None:
            features = self.activation(features)
        return features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution', None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, 'normalization', None) is not None:
            if hasattr(self.normalization, 'name'):
                with tf.name_scope(self.normalization.name):
                    self.normalization.build([None, None, None, self.out_channels])