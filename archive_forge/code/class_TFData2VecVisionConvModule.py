from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class TFData2VecVisionConvModule(keras.layers.Layer):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], padding: str='valid', bias: bool=False, dilation: Union[int, Tuple[int, int]]=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding=padding, use_bias=bias, dilation_rate=dilation, name='conv')
        self.bn = keras.layers.BatchNormalization(name='bn', momentum=0.9, epsilon=1e-05)
        self.activation = tf.nn.relu
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, input: tf.Tensor) -> tf.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv', None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, None, self.in_channels])
        if getattr(self, 'bn', None) is not None:
            with tf.name_scope(self.bn.name):
                self.bn.build((None, None, None, self.out_channels))