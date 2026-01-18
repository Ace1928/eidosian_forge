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
from .configuration_funnel import FunnelConfig
class TFFunnelPositionwiseFFN(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        initializer = get_initializer(config.initializer_range)
        self.linear_1 = keras.layers.Dense(config.d_inner, kernel_initializer=initializer, name='linear_1')
        self.activation_function = get_tf_activation(config.hidden_act)
        self.activation_dropout = keras.layers.Dropout(config.activation_dropout)
        self.linear_2 = keras.layers.Dense(config.d_model, kernel_initializer=initializer, name='linear_2')
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.config = config

    def call(self, hidden, training=False):
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h, training=training)
        h = self.linear_2(h)
        h = self.dropout(h, training=training)
        return self.layer_norm(hidden + h)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'linear_1', None) is not None:
            with tf.name_scope(self.linear_1.name):
                self.linear_1.build([None, None, self.config.d_model])
        if getattr(self, 'linear_2', None) is not None:
            with tf.name_scope(self.linear_2.name):
                self.linear_2.build([None, None, self.config.d_inner])
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])