from __future__ import annotations
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig
class TFDebertaContextPooler(keras.layers.Layer):

    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(config.pooler_hidden_size, name='dense')
        self.dropout = TFDebertaStableDropout(config.pooler_dropout, name='dropout')
        self.config = config

    def call(self, hidden_states, training: bool=False):
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token, training=training)
        pooled_output = self.dense(context_token)
        pooled_output = get_tf_activation(self.config.pooler_hidden_act)(pooled_output)
        return pooled_output

    @property
    def output_dim(self) -> int:
        return self.config.hidden_size

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense', None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.pooler_hidden_size])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)