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
class TFXLNetLMHead(keras.layers.Layer):

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.input_embeddings = input_embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)

    def get_output_embeddings(self):
        return self.input_embeddings

    def set_output_embeddings(self, value):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self):
        return {'bias': self.bias}

    def set_bias(self, value):
        self.bias = value['bias']
        self.config.vocab_size = shape_list(value['bias'])[0]

    def call(self, hidden_states):
        hidden_states = self.input_embeddings(hidden_states, mode='linear')
        hidden_states = hidden_states + self.bias
        return hidden_states