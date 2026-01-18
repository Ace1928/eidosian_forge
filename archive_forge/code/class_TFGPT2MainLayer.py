from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_gpt2 import GPT2Config
@keras_serializable
class TFGPT2MainLayer(keras.layers.Layer):
    config_class = GPT2Config

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict
        self.num_hidden_layers = config.n_layer
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range
        self.wte = keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='wte')
        self.wpe = keras.layers.Embedding(input_dim=config.n_positions, output_dim=config.n_embd, embeddings_initializer=get_initializer(config.initializer_range), name='wpe')
        self.drop = keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFBlock(config, scale=True, name=f'h_._{i}') for i in range(config.n_layer)]
        self.ln_f = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_f')
        self.embed_dim = config.hidden_size

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, encoder_hidden_states: np.ndarray | tf.Tensor | None=None, encoder_attention_mask: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = shape_list(past_key_values[0][0])[-2]
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(past_length, input_shape[-1] + past_length), axis=0)
        if attention_mask is not None:
            attention_mask_shape = shape_list(attention_mask)
            attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]))
            one_cst = tf.constant(1.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(tf.subtract(one_cst, attention_mask), tf.constant(-10000.0))
        if self.config.add_cross_attention and encoder_attention_mask is not None:
            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=encoder_hidden_states.dtype)
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None
        encoder_attention_mask = encoder_extended_attention_mask
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = tf.constant(0.0)
        position_embeds = tf.cast(position_embeds, dtype=inputs_embeds.dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)
        output_shape = input_shape + [shape_list(hidden_states)[-1]]
        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)
            outputs = block(hidden_states, layer_past, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, use_cache, output_attentions, training=training)
            hidden_states, present = outputs[:2]
            if use_cache:
                presents = presents + (present,)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if output_attentions:
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple((tf.reshape(t, attention_output_shape) for t in all_attentions))
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions, all_cross_attentions] if v is not None))
        return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions, cross_attentions=all_cross_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'wte', None) is not None:
            with tf.name_scope(self.wte.name):
                self.wte.build(None)
        if getattr(self, 'wpe', None) is not None:
            with tf.name_scope(self.wpe.name):
                self.wpe.build(None)
        if getattr(self, 'ln_f', None) is not None:
            with tf.name_scope(self.ln_f.name):
                self.ln_f.build([None, None, self.embed_dim])
        if getattr(self, 'h', None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)