from __future__ import annotations
import math
from typing import Optional, Tuple
import tensorflow as tf
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
from ...utils import add_start_docstrings_to_model_forward, logging
from .configuration_blip import BlipTextConfig
class TFBlipTextLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.attention = TFBlipTextAttention(config, name='attention')
        if self.config.is_decoder:
            self.crossattention = TFBlipTextAttention(config, is_cross_attention=self.config.is_decoder, name='crossattention')
        self.intermediate = TFBlipTextIntermediate(config, name='intermediate')
        self.self_output = TFBlipTextOutput(config, name='output')

    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, training=None):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, past_key_value=self_attn_past_key_value, training=training)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions, training=training)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.self_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + outputs
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
        if getattr(self, 'self_output', None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
        if getattr(self, 'crossattention', None) is not None:
            with tf.name_scope(self.crossattention.name):
                self.crossattention.build(None)