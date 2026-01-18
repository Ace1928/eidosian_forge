from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_roformer import RoFormerConfig
class TFRoFormerLayer(keras.layers.Layer):

    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFRoFormerAttention(config, name='attention')
        self.intermediate = TFRoFormerIntermediate(config, name='intermediate')
        self.roformer_output = TFRoFormerOutput(config, name='output')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, sinusoidal_pos: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        attention_outputs = self.attention(input_tensor=hidden_states, attention_mask=attention_mask, sinusoidal_pos=sinusoidal_pos, head_mask=head_mask, output_attentions=output_attentions, training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(hidden_states=attention_output)
        layer_output = self.roformer_output(hidden_states=intermediate_output, input_tensor=attention_output, training=training)
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
        if getattr(self, 'roformer_output', None) is not None:
            with tf.name_scope(self.roformer_output.name):
                self.roformer_output.build(None)