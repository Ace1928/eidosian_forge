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
class TFCvtSelfOutput(keras.layers.Layer):
    """Output of the Attention layer ."""

    def __init__(self, config: CvtConfig, embed_dim: int, drop_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = keras.layers.Dropout(drop_rate)
        self.embed_dim = embed_dim

    def call(self, hidden_state: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_state = self.dense(inputs=hidden_state)
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.embed_dim])