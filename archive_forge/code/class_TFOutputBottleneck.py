from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFOutputBottleneck(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(config.hidden_size, name='dense')
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size, epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.config = config

    def call(self, hidden_states, residual_tensor, training=False):
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs, training=training)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.true_hidden_size])
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)