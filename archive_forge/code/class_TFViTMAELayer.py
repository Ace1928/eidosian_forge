from __future__ import annotations
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_vit_mae import ViTMAEConfig
class TFViTMAELayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFViTMAEAttention(config, name='attention')
        self.intermediate = TFViTMAEIntermediate(config, name='intermediate')
        self.vit_output = TFViTMAEOutput(config, name='output')
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_before')
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_after')
        self.config = config

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(input_tensor=self.layernorm_before(inputs=hidden_states), head_mask=head_mask, output_attentions=output_attentions, training=training)
        attention_output = attention_outputs[0]
        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(inputs=hidden_states)
        intermediate_output = self.intermediate(hidden_states=layer_output)
        layer_output = self.vit_output(hidden_states=intermediate_output, input_tensor=hidden_states, training=training)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

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
        if getattr(self, 'vit_output', None) is not None:
            with tf.name_scope(self.vit_output.name):
                self.vit_output.build(None)
        if getattr(self, 'layernorm_before', None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])
        if getattr(self, 'layernorm_after', None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])