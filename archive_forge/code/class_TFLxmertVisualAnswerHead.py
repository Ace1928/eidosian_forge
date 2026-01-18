from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
class TFLxmertVisualAnswerHead(keras.layers.Layer):

    def __init__(self, config, num_labels, **kwargs):
        super().__init__(**kwargs)
        hid_dim = config.hidden_size
        self.dense = keras.layers.Dense(hid_dim * 2, kernel_initializer=get_initializer(config.initializer_range), name='logit_fc_._0')
        self.activation = get_tf_activation('gelu')
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='logit_fc_._2')
        self.dense_1 = keras.layers.Dense(num_labels, kernel_initializer=get_initializer(config.initializer_range), name='logit_fc_._3')
        self.hid_dim = hid_dim

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense_1(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.hid_dim])
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, self.hid_dim * 2])
        if getattr(self, 'dense_1', None) is not None:
            with tf.name_scope(self.dense_1.name):
                self.dense_1.build([None, None, self.hid_dim * 2])