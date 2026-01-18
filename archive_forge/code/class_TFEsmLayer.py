from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
class TFEsmLayer(keras.layers.Layer):

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TFEsmAttention(config, name='attention')
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f'{self} should be used as a decoder model if cross attention is added')
            self.crossattention = TFEsmAttention(config)
        self.intermediate = TFEsmIntermediate(config, name='intermediate')
        self.output_layer = TFEsmOutput(config, name='output')
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.config = config

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, training=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, training=training)
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise AttributeError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, cross_attn_past_key_value, output_attentions, training=training)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layernorm_output = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(hidden_states=layernorm_output)
        layer_output = self.output_layer(hidden_states=intermediate_output, input_tensor=attention_output, training=training)
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
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
        if getattr(self, 'output_layer', None) is not None:
            with tf.name_scope(self.output_layer.name):
                self.output_layer.build(None)
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])