from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_bart import BartConfig
class TFBartClassificationHead(keras.layers.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, inner_dim: int, num_classes: int, pooler_dropout: float, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = keras.layers.Dense(inner_dim, name='dense')
        self.dropout = keras.layers.Dropout(pooler_dropout)
        self.out_proj = keras.layers.Dense(num_classes, name='out_proj')
        self.input_dim = inner_dim
        self.inner_dim = inner_dim

    def call(self, inputs):
        hidden_states = self.dropout(inputs)
        hidden_states = self.dense(hidden_states)
        hidden_states = keras.activations.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.input_dim])
        if getattr(self, 'out_proj', None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.inner_dim])