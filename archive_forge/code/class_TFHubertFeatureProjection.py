from __future__ import annotations
import warnings
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_hubert import HubertConfig
class TFHubertFeatureProjection(keras.layers.Layer):

    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.projection = keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), bias_initializer='zeros', name='projection')
        self.dropout = keras.layers.Dropout(rate=config.feat_proj_dropout)
        self.config = config

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.conv_dim[-1]])
        if getattr(self, 'projection', None) is not None:
            with tf.name_scope(self.projection.name):
                self.projection.build([None, None, self.config.conv_dim[-1]])