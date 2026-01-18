from __future__ import annotations
import math
import random
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions, TFCausalLMOutputWithCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import logging
from .configuration_xglm import XGLMConfig
@keras_serializable
class TFXGLMMainLayer(keras.layers.Layer):
    config_class = XGLMConfig

    def __init__(self, config: XGLMConfig, embed_tokens: Optional[TFSharedEmbeddings]=None, *inputs, **kwargs: Any) -> None:
        super().__init__(*inputs, **kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = TFSharedEmbeddings(config.vocab_size, config.d_model, self.padding_idx, name='embed_tokens')
        self.offset = 2
        self._embed_positions_weights = create_sinusoidal_positions(num_positions=config.max_position_embeddings + self.offset, embedding_dim=config.d_model, padding_idx=config.pad_token_id)
        self.dropout = keras.layers.Dropout(config.dropout)
        self.layers = [TFXGLMDecoderLayer(config, name=f'layers.{i}') for i in range(config.num_layers)]
        self.layerdrop = config.layerdrop
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm')

    def get_input_embeddings(self) -> TFSharedEmbeddings:
        return self.embed_tokens

    def set_input_embeddings(self, value: TFSharedEmbeddings) -> None:
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask: tf.Tensor | None, input_shape: tf.TensorShape, past_key_values_length: int) -> tf.Tensor:
        combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length)
        combined_attention_mask = tf.cond(input_shape[-1] > 1, lambda: combined_attention_mask, lambda: tf.ones_like(combined_attention_mask))
        if attention_mask is None:
            return combined_attention_mask
        expand_attention_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
        return expand_attention_mask + combined_attention_mask

    def embed_positions(self, position_ids: np.ndarray | tf.Tensor | None=None) -> tf.Tensor:
        position_ids += self.offset
        positions = tf.gather(self._embed_positions_weights, position_ids, axis=0)
        return positions

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, cross_attn_head_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False, **kwargs: Any) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = tf.shape(input_ids)
            input_ids = tf.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = tf.shape(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(past_key_values_length, input_shape[-1] + past_key_values_length), axis=0)
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.embed_tokens.vocab_size)
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, tgt_len=input_shape[-1])
        positions = self.embed_positions(position_ids)
        hidden_states = tf.cast(inputs_embeds, dtype=tf.float32) + positions
        hidden_states = self.dropout(hidden_states, training=training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        next_decoder_cache = () if use_cache else None
        for attn_mask_name, attn_mask in [('head_mask', head_mask), ('cross_attn_head_mask', cross_attn_head_mask)]:
            if attn_mask is not None:
                tf.debugging.assert_equal(shape_list(attn_mask)[0], len(self.layers), message=f'The {attn_mask_name} should be specified for {len(self.layers)} layers, but it is for {shape_list(attn_mask)[0]}.')
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if training and dropout_probability < self.layerdrop:
                continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            hidden_states, layer_self_attn, layer_cross_attn, present_key_value = decoder_layer(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value)
            if use_cache:
                next_decoder_cache += (present_key_value,)
            if output_attentions:
                all_self_attns += (layer_self_attn,)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_cross_attn,)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])
        if getattr(self, 'embed_tokens', None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)