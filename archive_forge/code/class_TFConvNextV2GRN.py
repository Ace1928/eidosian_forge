from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import (
from .configuration_convnextv2 import ConvNextV2Config
class TFConvNextV2GRN(keras.layers.Layer):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, config: ConvNextV2Config, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape: tf.TensorShape=None):
        self.weight = self.add_weight(name='weight', shape=(1, 1, 1, self.dim), initializer=keras.initializers.Zeros())
        self.bias = self.add_weight(name='bias', shape=(1, 1, 1, self.dim), initializer=keras.initializers.Zeros())
        return super().build(input_shape)

    def call(self, hidden_states: tf.Tensor):
        global_features = tf.norm(hidden_states, ord='euclidean', axis=(1, 2), keepdims=True)
        norm_features = global_features / (tf.reduce_mean(global_features, axis=-1, keepdims=True) + 1e-06)
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        return hidden_states