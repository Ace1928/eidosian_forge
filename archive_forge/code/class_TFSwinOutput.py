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
class TFSwinOutput(keras.layers.Layer):

    def __init__(self, config: SwinConfig, dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(dim, name='dense')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob, 'dropout')
        self.config = config
        self.dim = dim

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, int(self.config.mlp_ratio * self.dim)])