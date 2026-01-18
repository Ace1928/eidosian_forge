from __future__ import annotations
import math
from typing import Optional, Tuple
import tensorflow as tf
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
from ...utils import add_start_docstrings_to_model_forward, logging
from .configuration_blip import BlipTextConfig
@keras_serializable
class TFBlipTextEncoder(keras.layers.Layer):
    config_class = BlipTextConfig

    def __init__(self, config, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.layer = [TFBlipTextLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    @unpack_inputs
    def call(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=None, output_attentions=False, output_hidden_states=False, return_dict=True, training=None):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None
        next_decoder_cache = () if use_cache else None
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)