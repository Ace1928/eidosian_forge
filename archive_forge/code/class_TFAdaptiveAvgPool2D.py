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
class TFAdaptiveAvgPool2D(keras.layers.Layer):

    def __init__(self, output_dims: Tuple[int, int], input_ordering: str='NHWC', **kwargs):
        super().__init__(**kwargs)
        self.output_dims = output_dims
        self.input_ordering = input_ordering
        if input_ordering not in ('NCHW', 'NHWC'):
            raise ValueError("Unrecognized input_ordering, should be 'NCHW' or 'NHWC'!")
        self.h_axis = input_ordering.index('H')
        self.w_axis = input_ordering.index('W')

    def pseudo_1d_pool(self, inputs: tf.Tensor, h_pooling: bool):
        if h_pooling:
            axis = self.h_axis
            output_dim = self.output_dims[0]
        else:
            axis = self.w_axis
            output_dim = self.output_dims[1]
        input_dim = inputs.shape[axis]
        small_window = math.ceil(input_dim / output_dim)
        big_window = small_window + 1
        if h_pooling:
            output_dim = self.output_dims[0]
            small_window_shape = (small_window, 1)
            big_window_shape = (big_window, 1)
        else:
            output_dim = self.output_dims[1]
            small_window_shape = (1, small_window)
            big_window_shape = (1, big_window)
        if output_dim == input_dim:
            return inputs
        elif output_dim == 1:
            return tf.reduce_mean(inputs, axis=axis, keepdims=True)
        elif input_dim % output_dim == 0:
            return tf.nn.avg_pool2d(inputs, ksize=small_window_shape, strides=small_window_shape, padding='VALID', data_format=self.input_ordering)
        elif output_dim > input_dim and output_dim % input_dim == 0:
            return tf.repeat(inputs, repeats=output_dim // input_dim, axis=axis)
        if output_dim < input_dim:
            small_pool = tf.nn.avg_pool2d(inputs, ksize=small_window_shape, strides=1, padding='VALID', data_format=self.input_ordering)
            big_pool = tf.nn.avg_pool2d(inputs, ksize=big_window_shape, strides=1, padding='VALID', data_format=self.input_ordering)
            both_pool = tf.concat([small_pool, big_pool], axis=axis)
        else:
            small_pool = inputs
            big_pool = tf.nn.avg_pool2d(inputs, ksize=big_window_shape, strides=1, padding='VALID', data_format=self.input_ordering)
            both_pool = tf.concat([small_pool, big_pool], axis=axis)
        window_starts = tf.math.floor(tf.range(output_dim, dtype=tf.float32) * input_dim / output_dim)
        window_starts = tf.cast(window_starts, tf.int64)
        window_ends = tf.math.ceil(tf.range(1, output_dim + 1, dtype=tf.float32) * input_dim / output_dim)
        window_ends = tf.cast(window_ends, tf.int64)
        pool_selector = tf.cast(window_ends - window_starts - small_window, tf.bool)
        small_indices = window_starts
        big_indices = window_starts + small_pool.shape[axis]
        gather_indices = tf.where(pool_selector, big_indices, small_indices)
        return tf.gather(both_pool, gather_indices, axis=axis)

    def call(self, inputs: tf.Tensor):
        if self.input_ordering == 'NHWC':
            input_shape = inputs.shape[1:3]
        else:
            input_shape = inputs.shape[2:]
        if self.output_dims[0] == self.output_dims[1] == 1:
            if self.input_ordering == 'NHWC':
                reduce_dims = [1, 2]
            else:
                reduce_dims = [2, 3]
            return tf.reduce_mean(inputs, axis=reduce_dims, keepdims=True)
        elif input_shape[0] % self.output_dims[0] == 0 and input_shape[1] % self.output_dims[1] == 0:
            h_resize = int(input_shape[0] // self.output_dims[0])
            w_resize = int(input_shape[1] // self.output_dims[1])
            return tf.nn.avg_pool2d(inputs, ksize=(h_resize, w_resize), strides=(h_resize, w_resize), padding='VALID', data_format=self.input_ordering)
        else:
            h_pooled = self.pseudo_1d_pool(inputs, h_pooling=True)
            return self.pseudo_1d_pool(h_pooled, h_pooling=False)