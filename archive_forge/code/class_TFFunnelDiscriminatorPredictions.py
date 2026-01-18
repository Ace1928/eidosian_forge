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
class TFFunnelDiscriminatorPredictions(keras.layers.Layer):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        initializer = get_initializer(config.initializer_range)
        self.dense = keras.layers.Dense(config.d_model, kernel_initializer=initializer, name='dense')
        self.activation_function = get_tf_activation(config.hidden_act)
        self.dense_prediction = keras.layers.Dense(1, kernel_initializer=initializer, name='dense_prediction')
        self.config = config

    def call(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation_function(hidden_states)
        logits = tf.squeeze(self.dense_prediction(hidden_states))
        return logits

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.d_model])
        if getattr(self, 'dense_prediction', None) is not None:
            with tf.name_scope(self.dense_prediction.name):
                self.dense_prediction.build([None, None, self.config.d_model])