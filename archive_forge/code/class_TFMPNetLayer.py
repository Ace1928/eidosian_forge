from __future__ import annotations
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mpnet import MPNetConfig
class TFMPNetLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFMPNetAttention(config, name='attention')
        self.intermediate = TFMPNetIntermediate(config, name='intermediate')
        self.out = TFMPNetOutput(config, name='output')

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions, position_bias=position_bias, training=training)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.out(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + outputs
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
        if getattr(self, 'out', None) is not None:
            with tf.name_scope(self.out.name):
                self.out.build(None)