from __future__ import annotations
import copy
import itertools
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_t5 import T5Config
class TFT5DenseGatedActDense(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        wi_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor * config.d_model ** (-0.5))
        wo_initializer = keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor * config.d_ff ** (-0.5))
        self.wi_0 = keras.layers.Dense(config.d_ff, use_bias=False, name='wi_0', kernel_initializer=wi_initializer)
        self.wi_1 = keras.layers.Dense(config.d_ff, use_bias=False, name='wi_1', kernel_initializer=wi_initializer)
        self.wo = keras.layers.Dense(config.d_model, use_bias=False, name='wo', kernel_initializer=wo_initializer)
        self.dropout = keras.layers.Dropout(config.dropout_rate)
        self.act = get_tf_activation(config.dense_act_fn)
        self.config = config

    def call(self, hidden_states, training=False):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.wo(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'wi_0', None) is not None:
            with tf.name_scope(self.wi_0.name):
                self.wi_0.build([None, None, self.config.d_model])
        if getattr(self, 'wi_1', None) is not None:
            with tf.name_scope(self.wi_1.name):
                self.wi_1.build([None, None, self.config.d_model])
        if getattr(self, 'wo', None) is not None:
            with tf.name_scope(self.wo.name):
                self.wo.build([None, None, self.config.d_ff])