from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_distilbert import DistilBertConfig
class TFFFN(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dropout = keras.layers.Dropout(config.dropout)
        self.lin1 = keras.layers.Dense(config.hidden_dim, kernel_initializer=get_initializer(config.initializer_range), name='lin1')
        self.lin2 = keras.layers.Dense(config.dim, kernel_initializer=get_initializer(config.initializer_range), name='lin2')
        self.activation = get_tf_activation(config.activation)
        self.config = config

    def call(self, input, training=False):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x, training=training)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'lin1', None) is not None:
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.config.dim])
        if getattr(self, 'lin2', None) is not None:
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.config.hidden_dim])