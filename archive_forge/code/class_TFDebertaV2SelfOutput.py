from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
class TFDebertaV2SelfOutput(keras.layers.Layer):

    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(config.hidden_size, name='dense')
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name='dropout')
        self.config = config

    def call(self, hidden_states, input_tensor, training: bool=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)