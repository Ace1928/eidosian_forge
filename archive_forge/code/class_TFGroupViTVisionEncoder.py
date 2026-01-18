from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class TFGroupViTVisionEncoder(keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stages = [TFGroupViTStage(config=config, depth=config.depths[i], num_group_token=config.num_group_tokens[i], num_output_group=config.num_output_groups[i], num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0, name=f'stages_._{i}') for i in range(len(config.depths))]

    def call(self, hidden_states: tf.Tensor, output_hidden_states: bool, output_attentions: bool, return_dict: bool, training: bool=False) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_groupings = () if output_attentions else None
        group_tokens = None
        for stage in self.stages:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = stage(hidden_states, group_tokens, output_attentions)
            hidden_states = layer_outputs[0]
            group_tokens = layer_outputs[1]
            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'stages', None) is not None:
            for layer in self.stages:
                with tf.name_scope(layer.name):
                    layer.build(None)