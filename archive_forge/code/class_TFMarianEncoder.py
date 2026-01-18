from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_marian import MarianConfig
@keras_serializable
class TFMarianEncoder(keras.layers.Layer):
    config_class = MarianConfig
    '\n    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a\n    [`TFMarianEncoderLayer`].\n\n    Args:\n        config: MarianConfig\n    '

    def __init__(self, config: MarianConfig, embed_tokens: Optional[keras.layers.Embedding]=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.dropout = keras.layers.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = TFMarianSinusoidalPositionalEmbedding(config.max_position_embeddings, config.d_model, name='embed_positions')
        self.layers = [TFMarianEncoderLayer(config, name=f'layers.{i}') for i in range(config.encoder_layers)]

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    def call(self, input_ids: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False):
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
            head_mask (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, `optional):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value
                in the config will be used instead.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail. This argument can be used only in eager mode, in graph mode the value in the config
                will be used instead.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used
                in eager mode, in graph mode the value will always be set to True.
            training (`bool`, *optional*, defaults to `False`):
                Whether or not to use the model in training mode (some modules like dropout modules have different
                behaviors between training and evaluation).
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.dropout(hidden_states, training=training)
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            tf.debugging.assert_equal(shape_list(head_mask)[0], len(self.layers), message=f'The head_mask should be specified for {len(self.layers)} layers, but it is for {shape_list(head_mask)[0]}.')
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if training and dropout_probability < self.layerdrop:
                continue
            hidden_states, attn = encoder_layer(hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None)
            if output_attentions:
                all_attentions += (attn,)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embed_positions', None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)