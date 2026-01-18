from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_opt import OPTConfig
@keras_serializable
class TFOPTDecoder(keras.layers.Layer):
    config_class = OPTConfig

    def __init__(self, config: OPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.layerdrop = config.layerdrop
        num_embeddings = config.max_position_embeddings
        self.embed_tokens = TFSharedEmbeddings(config.vocab_size, config.word_embed_proj_dim, config.pad_token_id, name='embed_tokens')
        self.embed_positions = TFOPTLearnedPositionalEmbedding(num_embeddings, config.hidden_size, name='embed_positions')
        if config.do_layer_norm_before and (not config._remove_final_layer_norm):
            self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-05, name='final_layer_norm')
        else:
            self.final_layer_norm = None
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = keras.layers.Dense(config.word_embed_proj_dim, name='project_out', use_bias=False)
            self.project_in = keras.layers.Dense(config.hidden_size, name='project_in', use_bias=False)
        else:
            self.project_in = None
            self.project_out = None
        self.layers = [TFOPTDecoderLayer(config, name=f'layers.{i}') for i in range(config.num_hidden_layers)]
        self.dropout = keras.layers.Dropout(config.dropout)

    def get_embed_tokens(self):
        return self.embed_tokens

    def set_embed_tokens(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens.vocab_size = new_embeddings.shape[0]
        self.embed_tokens.weight = new_embeddings

    def get_input_embeddings(self):
        return self.embed_tokens

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, past_key_values_length):
        _, seq_length = input_shape
        tf.debugging.assert_equal(seq_length + past_key_values_length, shape_list(attention_mask)[1], message=f'Attention mask shape should be (batch_size, seq_length + past_key_values_length) but is {shape_list(attention_mask)[1]} with input_ids shape {input_shape} and past length {past_key_values_length}.')
        expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(input_shape, past_key_values_length=past_key_values_length) + expanded_attn_mask
        else:
            combined_attention_mask = expanded_attn_mask
        return combined_attention_mask

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithPast, Tuple[tf.Tensor]]:
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

            head_mask (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`tf.Tensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            training (`bool`, *optional*, defaults to `False`):
                Whether or not to use the model in training mode (some modules like dropout modules have different
                behaviors between training and evaluation).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
            check_embeddings_within_bounds(input_ids, self.embed_tokens.vocab_size)
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = tf.ones((input_shape[0], input_shape[1] + past_key_values_length), dtype=tf.bool)
        else:
            tf.debugging.assert_equal(shape_list(attention_mask)[1], past_key_values_length + input_shape[1], message=f'The provided attention mask has length {tf.shape(attention_mask)[1]}, but its length should be {past_key_values_length + input_shape[1]} (sum of the lengths of current and past inputs)')
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length)
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        present_key_values = () if use_cache else None
        for attn_mask_name, attn_mask in [('head_mask', head_mask)]:
            if attn_mask is not None:
                tf.debugging.assert_equal(shape_list(attn_mask)[0], len(self.layers), message=f'The {attn_mask_name} should be specified for {len(self.layers)} layers, but it is for {shape_list(attn_mask)[0]}.')
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            hidden_states, layer_self_attn, present_key_value = decoder_layer(hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, past_key_value=past_key_value)
            if use_cache:
                present_key_values += (present_key_value,)
            if output_attentions:
                all_self_attns += (layer_self_attn,)
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, present_key_values, all_hidden_states, all_self_attns] if v is not None))
        else:
            return TFBaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=present_key_values, hidden_states=all_hidden_states, attentions=all_self_attns)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embed_tokens', None) is not None:
            with tf.name_scope(self.embed_tokens.name):
                self.embed_tokens.build(None)
        if getattr(self, 'embed_positions', None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, 'final_layer_norm', None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.config.hidden_size])
        if getattr(self, 'project_out', None) is not None:
            with tf.name_scope(self.project_out.name):
                self.project_out.build([None, None, self.config.hidden_size])
        if getattr(self, 'project_in', None) is not None:
            with tf.name_scope(self.project_in.name):
                self.project_in.build([None, None, self.config.word_embed_proj_dim])
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)