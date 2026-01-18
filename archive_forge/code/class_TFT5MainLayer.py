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
@keras_serializable
class TFT5MainLayer(keras.layers.Layer):
    config_class = T5Config

    def __init__(self, config, embed_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.config = config
        self.num_hidden_layers = config.num_layers
        self.block = [TFT5Block(config, has_relative_attention_bias=bool(i == 0), name=f'block_._{i}') for i in range(config.num_layers)]
        self.final_layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name='final_layer_norm')
        self.dropout = keras.layers.Dropout(config.dropout_rate)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, inputs_embeds=None, head_mask=None, encoder_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False) -> Tuple:
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(f'You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(f'You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds')
        if inputs_embeds is None:
            assert self.embed_tokens is not None, 'You have to initialize the model with valid token embeddings'
            check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = input_shape
        mask_seq_length = shape_list(past_key_values[0][0])[2] + seq_length if past_key_values is not None else seq_length
        if attention_mask is None:
            attention_mask = tf.fill((batch_size, mask_seq_length), 1)
        if self.is_decoder and encoder_attention_mask is None and (encoder_hidden_states is not None):
            encoder_seq_length = shape_list(encoder_hidden_states)[1]
            encoder_attention_mask = tf.fill((batch_size, encoder_seq_length), 1)
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        attention_mask = tf.cast(attention_mask, dtype=inputs_embeds.dtype)
        num_dims_attention_mask = len(shape_list(attention_mask))
        if num_dims_attention_mask == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif num_dims_attention_mask == 2:
            if self.is_decoder:
                seq_ids = tf.range(mask_seq_length)
                causal_mask = tf.less_equal(tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)), seq_ids[None, :, None])
                causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                if past_key_values[0] is not None:
                    extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000000.0
        if self.is_decoder and encoder_attention_mask is not None:
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=extended_attention_mask.dtype)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1000000000.0
        else:
            encoder_extended_attention_mask = None
        present_key_value_states = () if use_cache and self.is_decoder else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.is_decoder else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds, training=training)
        for idx, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask=extended_attention_mask, position_bias=position_bias, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, encoder_decoder_position_bias=encoder_decoder_position_bias, layer_head_mask=head_mask[idx] if head_mask is not None else None, encoder_layer_head_mask=encoder_head_mask[idx] if encoder_head_mask is not None else None, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions, training=training)
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if present_key_value_state is not None and use_cache and self.is_decoder:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            outputs = (hidden_states,)
            if use_cache and self.is_decoder:
                outputs = outputs + (present_key_value_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
                if self.is_decoder:
                    outputs + (all_cross_attentions,)
            return outputs
        if self.is_decoder:
            return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)
        else:
            return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'final_layer_norm', None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build(None)
        if getattr(self, 'block', None) is not None:
            for layer in self.block:
                with tf.name_scope(layer.name):
                    layer.build(None)