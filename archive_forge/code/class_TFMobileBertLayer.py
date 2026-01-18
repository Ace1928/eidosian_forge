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
class TFMobileBertLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks
        self.attention = TFMobileBertAttention(config, name='attention')
        self.intermediate = TFMobileBertIntermediate(config, name='intermediate')
        self.mobilebert_output = TFMobileBertOutput(config, name='output')
        if self.use_bottleneck:
            self.bottleneck = TFBottleneck(config, name='bottleneck')
        if config.num_feedforward_networks > 1:
            self.ffn = [TFFFNLayer(config, name=f'ffn.{i}') for i in range(config.num_feedforward_networks - 1)]

    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4
        attention_outputs = self.attention(query_tensor, key_tensor, value_tensor, layer_input, attention_mask, head_mask, output_attentions, training=training)
        attention_output = attention_outputs[0]
        s = (attention_output,)
        if self.num_feedforward_networks != 1:
            for i, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.mobilebert_output(intermediate_output, attention_output, hidden_states, training=training)
        outputs = (layer_output,) + attention_outputs[1:] + (tf.constant(0), query_tensor, key_tensor, value_tensor, layer_input, attention_output, intermediate_output) + s
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
        if getattr(self, 'mobilebert_output', None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)
        if getattr(self, 'bottleneck', None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        if getattr(self, 'ffn', None) is not None:
            for layer in self.ffn:
                with tf.name_scope(layer.name):
                    layer.build(None)