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
from .configuration_rembert import RemBertConfig
class TFRemBertEncoder(keras.layers.Layer):

    def __init__(self, config: RemBertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding_hidden_mapping_in = keras.layers.Dense(units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='embedding_hidden_mapping_in')
        self.layer = [TFRemBertLayer(config, name='layer_._{}'.format(i)) for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_values: Tuple[Tuple[tf.Tensor]], use_cache: bool, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        hidden_states = self.embedding_hidden_mapping_in(inputs=hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask[i], encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_decoder_cache, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embedding_hidden_mapping_in', None) is not None:
            with tf.name_scope(self.embedding_hidden_mapping_in.name):
                self.embedding_hidden_mapping_in.build([None, None, self.config.input_embedding_size])
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)