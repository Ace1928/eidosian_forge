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
class TFSwinEncoder(keras.layers.Layer):

    def __init__(self, config: SwinConfig, grid_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.num_layers = len(config.depths)
        self.config = config
        dpr = list((tf.linspace(0, 1, sum(config.depths)) * config.drop_path_rate).numpy())
        self.layers = [TFSwinStage(config=config, dim=int(config.embed_dim * 2 ** i_layer), input_resolution=(grid_size[0] // 2 ** i_layer, grid_size[1] // 2 ** i_layer), depth=config.depths[i_layer], num_heads=config.num_heads[i_layer], drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])], downsample=TFSwinPatchMerging if i_layer < self.num_layers - 1 else None, name=f'layers.{i_layer}') for i_layer in range(self.num_layers)]
        self.gradient_checkpointing = False

    def call(self, hidden_states: tf.Tensor, input_dimensions: Tuple[int, int], head_mask: tf.Tensor | None=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, training: bool=False) -> Union[Tuple[tf.Tensor, ...], TFSwinEncoderOutput]:
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            batch_size, _, hidden_size = shape_list(hidden_states)
            reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
            reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, training=training)
            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
            if output_hidden_states:
                batch_size, _, hidden_size = shape_list(hidden_states)
                reshaped_hidden_state = tf.reshape(hidden_states, (batch_size, *input_dimensions, hidden_size))
                reshaped_hidden_state = tf.transpose(reshaped_hidden_state, (0, 3, 1, 2))
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            if output_attentions:
                all_self_attentions += layer_outputs[2:]
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFSwinEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions, reshaped_hidden_states=all_reshaped_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)