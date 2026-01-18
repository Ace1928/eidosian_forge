from __future__ import annotations
import copy
import itertools
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_t5 import T5Config
class TFT5Block(keras.layers.Layer):

    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.is_decoder = config.is_decoder
        self.layer = []
        self.layer.append(TFT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, name='layer_._0'))
        if self.is_decoder:
            self.layer.append(TFT5LayerCrossAttention(config, name='layer_._1'))
        self.layer.append(TFT5LayerFF(config, name=f'layer_._{len(self.layer)}'))

    def call(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, layer_head_mask=None, encoder_layer_head_mask=None, past_key_value=None, use_cache=False, output_attentions=False, training=False):
        if past_key_value is not None:
            assert self.is_decoder, 'Only decoder can use `past_key_values`'
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4
            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(f'There should be {expected_num_past_key_values} past states. {('2 (past / key) for cross attention' if expected_num_past_key_values == 4 else '')}. Got {len(past_key_value)} past key / value states')
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = (None, None)
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, layer_head_mask=layer_head_mask, past_key_value=self_attn_past_key_value, use_cache=use_cache, output_attentions=output_attentions, training=training)
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]
        if self.is_decoder and encoder_hidden_states is not None:
            if present_key_value_state is not None:
                query_length = shape_list(present_key_value_state[0])[2]
            else:
                query_length = None
            cross_attention_outputs = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, layer_head_mask=encoder_layer_head_mask, past_key_value=cross_attn_past_key_value, query_length=query_length, use_cache=use_cache, output_attentions=output_attentions, training=training)
            hidden_states = cross_attention_outputs[0]
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]
            attention_outputs = attention_outputs + cross_attention_outputs[2:]
        hidden_states = self.layer[-1](hidden_states, training=training)
        outputs = (hidden_states,)
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        for layer_module in self.layer:
            if hasattr(layer_module, 'name'):
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)