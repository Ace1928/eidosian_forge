from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_deit import DeiTConfig
class TFDeitDecoder(keras.layers.Layer):

    def __init__(self, config: DeiTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv2d = keras.layers.Conv2D(filters=config.encoder_stride ** 2 * config.num_channels, kernel_size=1, name='0')
        self.pixel_shuffle = TFDeitPixelShuffle(config.encoder_stride, name='1')
        self.config = config

    def call(self, inputs: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_states = inputs
        hidden_states = self.conv2d(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv2d', None) is not None:
            with tf.name_scope(self.conv2d.name):
                self.conv2d.build([None, None, None, self.config.hidden_size])
        if getattr(self, 'pixel_shuffle', None) is not None:
            with tf.name_scope(self.pixel_shuffle.name):
                self.pixel_shuffle.build(None)