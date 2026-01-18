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
class TFCvtOutput(keras.layers.Layer):
    """
    Output of the Convolutional Transformer Block (last chunk). It consists of a MLP and a residual connection.
    """

    def __init__(self, config: CvtConfig, embed_dim: int, mlp_ratio: int, drop_rate: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(units=embed_dim, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.dropout = keras.layers.Dropout(drop_rate)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

    def call(self, hidden_state: tf.Tensor, input_tensor: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_state = self.dense(inputs=hidden_state)
        hidden_state = self.dropout(inputs=hidden_state, training=training)
        hidden_state = hidden_state + input_tensor
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, int(self.embed_dim * self.mlp_ratio)])