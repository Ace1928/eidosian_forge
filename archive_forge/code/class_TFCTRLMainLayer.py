from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig
@keras_serializable
class TFCTRLMainLayer(keras.layers.Layer):
    config_class = CTRLConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer
        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size)
        self.w = keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.n_embd, embeddings_initializer=get_initializer(config.initializer_range), name='w')
        self.dropout = keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFEncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop, config.layer_norm_epsilon, self.output_attentions, name=f'h_._{i}') for i in range(config.n_layer)]
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='layernorm')

    def get_input_embeddings(self):
        return self.w

    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, attention_mask: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFBaseModelOutputWithPast]:
        if past_key_values is not None:
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]
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
            position_ids = tf.expand_dims(tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32), axis=0)
            position_ids = tf.tile(position_ids, [input_shape[0], 1])
        if attention_mask is not None:
            attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1] + past_length))
            one_cst = tf.constant(1.0)
            ten_thousand_cst = tf.constant(-10000.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(tf.subtract(one_cst, attention_mask), ten_thousand_cst)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_layers
        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.w(token_type_ids)
            token_type_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, dtype=token_type_embeds.dtype))
        else:
            token_type_embeds = tf.constant(0.0)
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.w.input_dim)
            inputs_embeds = self.w(input_ids)
        seq_len = input_shape[-1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        inputs_embeds *= tf.math.sqrt(tf.cast(self.d_model_size, inputs_embeds.dtype))
        pos_embeds = tf.gather(self.pos_encoding, position_ids)
        pos_embeds = tf.cast(pos_embeds, dtype=token_type_embeds.dtype)
        hidden_states = inputs_embeds + pos_embeds + token_type_embeds
        hidden_states = self.dropout(hidden_states, training=training)
        output_shape = input_shape + [shape_list(hidden_states)[-1]]
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, (h, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)
            outputs = h(hidden_states, mask, layer_past, attention_mask, head_mask[i], use_cache, output_attentions, training=training)
            hidden_states, present = outputs[:2]
            if use_cache:
                presents = presents + (present,)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if output_attentions:
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple((tf.reshape(t, attention_output_shape) for t in all_attentions))
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'w', None) is not None:
            with tf.name_scope(self.w.name):
                self.w.build(None)
        if getattr(self, 'layernorm', None) is not None:
            with tf.name_scope(self.layernorm.name):
                self.layernorm.build([None, None, self.config.n_embd])
        if getattr(self, 'h', None) is not None:
            for layer in self.h:
                with tf.name_scope(layer.name):
                    layer.build(None)