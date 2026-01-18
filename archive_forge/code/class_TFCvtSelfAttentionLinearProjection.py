from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
class TFCvtSelfAttentionLinearProjection(keras.layers.Layer):
    """Linear projection layer used to flatten tokens into 1D."""

    def call(self, hidden_state: tf.Tensor) -> tf.Tensor:
        batch_size, height, width, num_channels = shape_list(hidden_state)
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))
        return hidden_state