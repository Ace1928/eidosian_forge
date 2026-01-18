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
class TFGroupViTMLP(keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, hidden_size: Optional[int]=None, intermediate_size: Optional[int]=None, output_size: Optional[int]=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.activation_fn = get_tf_activation(config.hidden_act)
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        output_size = output_size if output_size is not None else hidden_size
        self.fc1 = keras.layers.Dense(intermediate_size, name='fc1')
        self.fc2 = keras.layers.Dense(output_size, name='fc2')
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'fc1', None) is not None:
            with tf.name_scope(self.fc1.name):
                self.fc1.build([None, None, self.hidden_size])
        if getattr(self, 'fc2', None) is not None:
            with tf.name_scope(self.fc2.name):
                self.fc2.build([None, None, self.intermediate_size])