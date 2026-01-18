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
class TFMobileBertOutput(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.dense = keras.layers.Dense(config.true_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, epsilon=config.layer_norm_eps, name='LayerNorm')
        if not self.use_bottleneck:
            self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = TFOutputBottleneck(config, name='bottleneck')
        self.config = config

    def call(self, hidden_states, residual_tensor_1, residual_tensor_2, training=False):
        hidden_states = self.dense(hidden_states)
        if not self.use_bottleneck:
            hidden_states = self.dropout(hidden_states, training=training)
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
        else:
            hidden_states = self.LayerNorm(hidden_states + residual_tensor_1)
            hidden_states = self.bottleneck(hidden_states, residual_tensor_2)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)
        if getattr(self, 'bottleneck', None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)