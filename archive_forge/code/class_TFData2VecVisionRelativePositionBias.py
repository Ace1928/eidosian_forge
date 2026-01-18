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
class TFData2VecVisionRelativePositionBias(keras.layers.Layer):

    def __init__(self, config: Data2VecVisionConfig, window_size: tuple, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_index = self.get_position_index()

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(shape=(self.num_relative_distance, self.config.num_attention_heads), initializer='zeros', trainable=True, name='relative_position_bias_table')
        super().build(input_shape)

    def get_position_index(self):
        xx, yy = tf.meshgrid(range(self.window_size[0]), range(self.window_size[1]))
        coords = tf.stack([yy, xx], axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        xx = (relative_coords[:, :, 0] + self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        yy = relative_coords[:, :, 1] + self.window_size[1] - 1
        relative_coords = tf.stack([xx, yy], axis=-1)
        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)
        top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (self.num_relative_distance - 3)
        left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (self.num_relative_distance - 2)
        corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (self.num_relative_distance - 1)
        left_corner = tf.concat([corner, left], axis=0)
        relative_position_index = tf.concat([top, relative_position_index], axis=0)
        relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)
        return relative_position_index

    def call(self, inputs=None) -> tf.Tensor:
        relative_position_bias = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=0)
        return tf.transpose(relative_position_bias, [2, 0, 1])