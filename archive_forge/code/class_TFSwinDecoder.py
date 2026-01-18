from __future__ import annotations
import collections.abc
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_swin import SwinConfig
class TFSwinDecoder(keras.layers.Layer):

    def __init__(self, config: SwinConfig, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = keras.layers.Conv2D(filters=config.encoder_stride ** 2 * config.num_channels, kernel_size=1, strides=1, name='0')
        self.pixel_shuffle = TFSwinPixelShuffle(config.encoder_stride, name='1')
        self.config = config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        hidden_states = x
        hidden_states = tf.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv2d(hidden_states)
        hidden_states = self.pixel_shuffle(hidden_states)
        hidden_states = tf.transpose(hidden_states, (0, 3, 1, 2))
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