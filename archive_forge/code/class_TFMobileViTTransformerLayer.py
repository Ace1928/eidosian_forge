from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTTransformerLayer(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = TFMobileViTAttention(config, hidden_size, name='attention')
        self.intermediate = TFMobileViTIntermediate(config, hidden_size, intermediate_size, name='intermediate')
        self.mobilevit_output = TFMobileViTOutput(config, hidden_size, intermediate_size, name='output')
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_before')
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_after')
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, training: bool=False) -> tf.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_states), training=training)
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.mobilevit_output(layer_output, hidden_states, training=training)
        return layer_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'intermediate', None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, 'mobilevit_output', None) is not None:
            with tf.name_scope(self.mobilevit_output.name):
                self.mobilevit_output.build(None)
        if getattr(self, 'layernorm_before', None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.hidden_size])
        if getattr(self, 'layernorm_after', None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.hidden_size])