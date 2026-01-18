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
from .configuration_mobilebert import MobileBertConfig
class TFBottleneck(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.bottleneck_input = TFBottleneckLayer(config, name='input')
        if self.key_query_shared_bottleneck:
            self.attention = TFBottleneckLayer(config, name='attention')

    def call(self, hidden_states):
        bottlenecked_hidden_states = self.bottleneck_input(hidden_states)
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        else:
            return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'bottleneck_input', None) is not None:
            with tf.name_scope(self.bottleneck_input.name):
                self.bottleneck_input.build(None)
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)