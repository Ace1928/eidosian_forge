import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class TFEfficientFormerFlat(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor]:
        batch_size, _, _, in_channels = shape_list(hidden_states)
        hidden_states = tf.reshape(hidden_states, shape=[batch_size, -1, in_channels])
        return hidden_states