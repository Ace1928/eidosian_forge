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
class TFConvNextV2Layer(keras.layers.Layer):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch. Since we already permuted the inputs to follow
    NHWC ordering, we can just apply the operations straight-away without the permutation.

    Args:
        config (`ConvNextV2Config`):
            Model configuration class.
        dim (`int`):
            Number of input channels.
        drop_path (`float`, defaults to 0.0):
            Stochastic depth rate.
    """

    def __init__(self, config: ConvNextV2Config, dim: int, drop_path: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.config = config
        self.dwconv = keras.layers.Conv2D(filters=dim, kernel_size=7, padding='same', groups=dim, kernel_initializer=get_initializer(config.initializer_range), bias_initializer=keras.initializers.Zeros(), name='dwconv')
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-06, name='layernorm')
        self.pwconv1 = keras.layers.Dense(units=4 * dim, kernel_initializer=get_initializer(config.initializer_range), bias_initializer=keras.initializers.Zeros(), name='pwconv1')
        self.act = get_tf_activation(config.hidden_act)
        self.grn = TFConvNextV2GRN(config, 4 * dim, dtype=tf.float32, name='grn')
        self.pwconv2 = keras.layers.Dense(units=dim, kernel_initializer=get_initializer(config.initializer_range), bias_initializer=keras.initializers.Zeros(), name='pwconv2')
        self.drop_path = TFConvNextV2DropPath(drop_path, name='drop_path') if drop_path > 0.0 else keras.layers.Activation('linear', name='drop_path')

    def call(self, hidden_states, training=False):
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.drop_path(x, training=training)
        x = input + x
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dwconv', None) is not None:
            with tf.name_scope(self.dwconv.name):
                self.dwconv.build([None, None, None, self.dim])
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, None, self.dim])
        if getattr(self, 'pwconv1', None) is not None:
            with tf.name_scope(self.pwconv1.name):
                self.pwconv1.build([None, None, self.dim])
        if getattr(self, 'grn', None) is not None:
            with tf.name_scope(self.grn.name):
                self.grn.build(None)
        if getattr(self, 'pwconv2', None) is not None:
            with tf.name_scope(self.pwconv2.name):
                self.pwconv2.build([None, None, 4 * self.dim])
        if getattr(self, 'drop_path', None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)