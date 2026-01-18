from __future__ import annotations
import random
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_blenderbot_small import BlenderbotSmallConfig
@keras_serializable
class TFBlenderbotSmallDecoder(keras.layers.Layer):
    config_class = BlenderbotSmallConfig
    '\n    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`TFBlenderbotSmallDecoderLayer`]\n\n    Args:\n        config: BlenderbotSmallConfig\n        embed_tokens: output embedding\n    '

    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[keras.layers.Embedding]=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = embed_tokens
        self.layerdrop = config.decoder_layerdrop
        self.embed_positions = TFBlenderbotSmallLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model, name='embed_positions')
        self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        self.layers = [TFBlenderbotSmallDecoderLayer(config, name=f'layers.{i}') for i in range(config.decoder_layers)]
        self.layernorm_embedding = keras.layers.LayerNormalization(epsilon=1e-05, name='layernorm_embedding')
        self.dropout = keras.layers.Dropout(config.dropout)

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    @unpack_inputs
    def call(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, encoder_hidden_states=None, encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
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
            position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the
                range `[0, config.max_position_embeddings - 1]`.
            encoder_hidden_states (`tf.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`tf.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
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
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        past_key_values_length = shape_list(past_key_values[0][0])[2] if past_key_values is not None else 0
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.embed_tokens.input_dim)
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length)
        else:
            combined_attention_mask = _expand_mask(tf.ones((input_shape[0], input_shape[1] + past_key_values_length)), tgt_len=input_shape[-1])
        if attention_mask is not None:
            combined_attention_mask = combined_attention_mask + _expand_mask(attention_mask, tgt_len=input_shape[-1])
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, tgt_len=input_shape[-1])
        if position_ids is None:
            positions = self.embed_positions(input_shape, past_key_values_length)
        else:
            positions = self.embed_positions(input_shape, position_ids=position_ids)
        hidden_states = self.layernorm_embedding(inputs_embeds) + positions
        hidden_states = self.dropout(hidden_states, training=training)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions and encoder_hidden_states is not None else None
        present_key_values = () if use_cache else None
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
            hidden_states, layer_self_attn, layer_cross_attn, present_key_value = decoder_layer(hidden_states, attention_mask=combined_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value)
            if use_cache:
                present_key_values += (present_key_value,)
            if output_attentions:
                all_self_attns += (layer_self_attn,)
                if encoder_hidden_states is not None:
                    all_cross_attns += (layer_cross_attn,)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return (hidden_states, present_key_values, all_hidden_states, all_self_attns, all_cross_attns)
        else:
            return TFBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_values, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attns)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embed_positions', None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, 'layernorm_embedding', None) is not None:
            with tf.name_scope(self.layernorm_embedding.name):
                self.layernorm_embedding.build([None, None, self.config.d_model])
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)