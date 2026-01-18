from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_led import LEDConfig
@keras_serializable
class TFLEDEncoder(keras.layers.Layer):
    config_class = LEDConfig
    '\n    Transformer encoder consisting of *config.encoder_layers* self-attention layers. Each layer is a\n    [`TFLEDEncoderLayer`].\n\n    Args:\n        config: LEDConfig\n    '

    def __init__(self, config: LEDConfig, embed_tokens: Optional[keras.layers.Embedding]=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)
        if config.encoder_layerdrop > 0:
            logger.warning('Layerdrop is currently disabled in TFLED models.')
        self.layerdrop = 0.0
        self.padding_idx = config.pad_token_id
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, '`config.attention_window` has to be an even value'
            assert config.attention_window > 0, '`config.attention_window` has to be positive'
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        else:
            assert len(config.attention_window) == config.num_hidden_layers, f'`len(config.attention_window)` should equal `config.num_hidden_layers`. Expected {config.num_hidden_layers}, given {len(config.attention_window)}'
        self.attention_window = config.attention_window
        self.embed_tokens = embed_tokens
        self.embed_positions = TFLEDLearnedPositionalEmbedding(config.max_encoder_position_embeddings, config.d_model, name='embed_positions')
        self.layers = [TFLEDEncoderLayer(config, i, name=f'layers.{i}') for i in range(config.encoder_layers)]
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-05, name='layernorm_embedding')
        self.embed_dim = config.d_model

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    def call(self, input_ids=None, inputs_embeds=None, attention_mask=None, global_attention_mask=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        """
        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if global_attention_mask is not None:
            attention_mask = attention_mask * tf.cast(global_attention_mask + 1, dtype=attention_mask.dtype)
        padding_len, input_ids, attention_mask, inputs_embeds = self._pad_to_window_size(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, pad_token_id=self.padding_idx)
        input_shape = shape_list(attention_mask)
        is_index_masked = tf.math.less(tf.cast(attention_mask, tf.int8), 1)
        is_index_global_attn = tf.math.greater(tf.cast(attention_mask, tf.int8), 1)
        is_global_attn = tf.math.reduce_any(is_index_global_attn)
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask)[:, 0, 0, :]
            attention_mask = attention_mask[:, :, None, None]
        encoder_states = () if output_hidden_states else None
        all_attentions = all_global_attentions = () if output_attentions else None
        if head_mask is not None:
            tf.debugging.assert_equal(shape_list(head_mask)[0], len(self.layers), message=f'The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(head_mask)[0]}.')
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                hidden_states_to_add = self.compute_hidden_states(hidden_states, padding_len)
                encoder_states = encoder_states + (hidden_states_to_add,)
            dropout_probability = random.uniform(0, 1)
            if training and dropout_probability < self.layerdrop:
                continue
            layer_outputs = encoder_layer(hidden_states=hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (tf.transpose(layer_outputs[1], (0, 2, 1, 3)),)
                all_global_attentions = all_global_attentions + (tf.transpose(layer_outputs[2], (0, 1, 3, 2)),)
        hidden_states = self.compute_hidden_states(hidden_states, padding_len)
        if output_attentions:
            all_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions]) if padding_len > 0 else all_attentions
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return TFLEDEncoderBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, global_attentions=all_global_attentions)

    @tf.function
    def compute_hidden_states(self, hidden_states, padding_len):
        return hidden_states[:, :-padding_len] if padding_len > 0 else hidden_states

    def _pad_to_window_size(self, input_ids, attention_mask, inputs_embeds, pad_token_id):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""
        attention_window = self.attention_window if isinstance(self.attention_window, int) else max(self.attention_window)
        assert attention_window % 2 == 0, f'`attention_window` should be an even value. Given {attention_window}'
        input_shape = shape_list(input_ids) if input_ids is not None else shape_list(inputs_embeds)
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.warning_once(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.attention_window`: {attention_window}')
        paddings = tf.convert_to_tensor([[0, 0], [0, padding_len]])
        if input_ids is not None:
            input_ids = tf.pad(input_ids, paddings, constant_values=pad_token_id)
        if inputs_embeds is not None:
            if padding_len > 0:
                input_ids_padding = tf.fill((batch_size, padding_len), pad_token_id)
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                inputs_embeds = tf.concat([inputs_embeds, inputs_embeds_padding], axis=-2)
        attention_mask = tf.pad(attention_mask, paddings, constant_values=False)
        return (padding_len, input_ids, attention_mask, inputs_embeds)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embed_positions', None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, 'layernorm_embedding', None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.embed_dim])
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)