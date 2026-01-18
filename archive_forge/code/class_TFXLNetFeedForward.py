from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
class TFXLNetFeedForward(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.layer_1 = keras.layers.Dense(config.d_inner, kernel_initializer=get_initializer(config.initializer_range), name='layer_1')
        self.layer_2 = keras.layers.Dense(config.d_model, kernel_initializer=get_initializer(config.initializer_range), name='layer_2')
        self.dropout = keras.layers.Dropout(config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = get_tf_activation(config.ff_activation)
        else:
            self.activation_function = config.ff_activation
        self.config = config

    def call(self, inp, training=False):
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output, training=training)
        output = self.layer_2(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output + inp)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, 'layer_1', None) is not None:
            with tf.name_scope(self.layer_1.name):
                self.layer_1.build([None, None, self.config.d_model])
        if getattr(self, 'layer_2', None) is not None:
            with tf.name_scope(self.layer_2.name):
                self.layer_2.build([None, None, self.config.d_inner])