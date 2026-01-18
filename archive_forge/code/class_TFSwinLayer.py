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
class TFSwinLayer(keras.layers.Layer):

    def __init__(self, config, dim, input_resolution: Tuple[int, int], num_heads: int, shift_size: int=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        min_res = tf.reduce_min(input_resolution)
        self.window_size = min_res if min_res <= config.window_size else config.window_size
        self.shift_size = 0 if min_res <= self.window_size else shift_size
        self.input_resolution = input_resolution
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_before')
        self.attention = TFSwinAttention(config, dim, num_heads, name='attention')
        self.drop_path = TFSwinDropPath(config.drop_path_rate, name='drop_path') if config.drop_path_rate > 0.0 else keras.layers.Activation('linear', name='drop_path')
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_after')
        self.intermediate = TFSwinIntermediate(config, dim, name='intermediate')
        self.swin_output = TFSwinOutput(config, dim, name='output')
        self.dim = dim

    def get_attn_mask(self, height: int, width: int, window_size: int, shift_size: int) -> tf.Tensor | None:
        img_mask = tf.zeros((height, width))
        height_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))
        width_slices = ((0, -window_size), (-window_size, -shift_size), (-shift_size, -1))
        if shift_size > 0:
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    height_inds = tf.range(height_slice[0] % height, height_slice[1] % height + 1)
                    width_inds = tf.range(width_slice[0] % width, width_slice[1] % width + 1)
                    indices = tf.reshape(tf.stack(tf.meshgrid(height_inds, width_inds), axis=-1), (-1, 2))
                    if len(indices) >= 1:
                        updates = tf.ones((len(indices),), dtype=img_mask.dtype) * count
                        img_mask = tf.tensor_scatter_nd_update(img_mask, indices, updates)
                    count += 1
        img_mask = tf.expand_dims(img_mask, -1)
        img_mask = tf.expand_dims(img_mask, 0)
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = tf.reshape(mask_windows, (-1, window_size * window_size))
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, float(-100.0), attn_mask)
        attn_mask = tf.where(attn_mask == 0, float(0.0), attn_mask)
        return attn_mask

    def maybe_pad(self, hidden_states: tf.Tensor, window_size: int, height: int, width: int) -> Tuple[tf.Tensor, tf.Tensor]:
        pad_right = (window_size - width % window_size) % window_size
        pad_bottom = (window_size - height % window_size) % window_size
        pad_values = [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]]
        hidden_states = tf.pad(hidden_states, pad_values)
        pad_values = tf.reshape(pad_values, (-1,))
        return (hidden_states, pad_values)

    def call(self, hidden_states: tf.Tensor, input_dimensions: Tuple[int, int], head_mask: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> tf.Tensor:
        min_res = tf.reduce_min(input_dimensions)
        shift_size = 0 if min_res <= self.window_size else self.shift_size
        window_size = min_res if min_res <= self.window_size else self.window_size
        height, width = input_dimensions
        batch_size, _, channels = shape_list(hidden_states)
        shortcut = hidden_states
        hidden_states = self.layernorm_before(hidden_states, training=training)
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, channels))
        hidden_states, pad_values = self.maybe_pad(hidden_states, window_size, height, width)
        _, height_pad, width_pad, _ = shape_list(hidden_states)
        if shift_size > 0:
            shifted_hidden_states = tf.roll(hidden_states, shift=(-shift_size, -shift_size), axis=(1, 2))
        else:
            shifted_hidden_states = hidden_states
        hidden_states_windows = window_partition(shifted_hidden_states, window_size)
        hidden_states_windows = tf.reshape(hidden_states_windows, (-1, window_size * window_size, channels))
        attn_mask = self.get_attn_mask(height=height_pad, width=width_pad, window_size=window_size, shift_size=shift_size)
        attention_outputs = self.attention(hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions, training=training)
        attention_output = attention_outputs[0]
        attention_windows = tf.reshape(attention_output, (-1, window_size, window_size, channels))
        shifted_windows = window_reverse(attention_windows, window_size, height_pad, width_pad)
        if shift_size > 0:
            attention_windows = tf.roll(shifted_windows, shift=(shift_size, shift_size), axis=(1, 2))
        else:
            attention_windows = shifted_windows
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]
        attention_windows = tf.reshape(attention_windows, (batch_size, height * width, channels))
        hidden_states = shortcut + self.drop_path(attention_windows, training=training)
        layer_output = self.layernorm_after(hidden_states, training=training)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.swin_output(layer_output, training=training)
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layernorm_before', None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.dim])
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'drop_path', None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)
        if getattr(self, 'layernorm_after', None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.dim])
        if getattr(self, 'intermediate', None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, 'swin_output', None) is not None:
            with tf.name_scope(self.swin_output.name):
                self.swin_output.build(None)