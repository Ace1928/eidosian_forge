from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
class TFLxmertLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFLxmertSelfAttentionLayer(config, name='attention')
        self.intermediate = TFLxmertIntermediate(config, name='intermediate')
        self.transformer_output = TFLxmertOutput(config, name='output')

    def call(self, hidden_states, attention_mask, output_attentions, training=False):
        attention_outputs = self.attention(hidden_states, attention_mask, output_attentions, training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.transformer_output(intermediate_output, attention_output, training=training)
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
        if getattr(self, 'transformer_output', None) is not None:
            with tf.name_scope(self.transformer_output.name):
                self.transformer_output.build(None)