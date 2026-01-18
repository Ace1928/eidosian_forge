import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
@add_start_docstrings('The standalone decoder part of the XLMProphetNetModel.', XLM_PROPHETNET_START_DOCSTRING)
class XLMProphetNetDecoder(XLMProphetNetPreTrainedModel):
    """
    word_embeddings  (`torch.nn.Embeddings` of shape `(config.vocab_size, config.hidden_size)`, *optional*):
        The word embedding parameters. This can be used to initialize [`XLMProphetNetEncoder`] with pre-defined word
        embeddings instead of randomly initialized word embeddings.
    """

    def __init__(self, config: XLMProphetNetConfig, word_embeddings: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.ngram = config.ngram
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.dropout = config.dropout
        self.max_target_positions = config.max_position_embeddings
        self.word_embeddings = word_embeddings if word_embeddings is not None else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)
        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size, None)
        self.layers = nn.ModuleList([XLMProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.embeddings_layer_norm = LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(XLM_PROPHETNET_STANDALONE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XLMProphetNetDecoderModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, XLMProphetNetDecoderModelOutput]:
        """
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, XLMProphetNetDecoder
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone")
        >>> model = XLMProphetNetDecoder.from_pretrained("patrickvonplaten/xprophetnet-large-uncased-standalone", add_cross_attention=False)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Either `decoder_input_ids` or `decoder_inputs_embeds` has to be passed.')
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError('Make sure to only pass `decoder_input_ids` or `decoder_inputs_embeds`.')
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        batch_size, sequence_length = inputs_embeds.shape[:2]
        main_stream_pos_embed, position_ids = self.position_embeddings((batch_size, sequence_length), device=inputs_embeds.device, past_key_values=past_key_values)
        if past_key_values is not None:
            main_relative_position_buckets, predict_relative_position_buckets = (None, None)
        else:
            main_relative_position_buckets, predict_relative_position_buckets = self.compute_buffered_relative_buckets(position_ids)
        predicting_stream_pos_embed = self.position_embeddings._forward(position_ids + 1)
        hidden_states = inputs_embeds + main_stream_pos_embed
        ngram_embeddings = self.ngram_embeddings.weight
        if past_key_values is not None:
            assert hidden_states.size(1) == 1, 'At the moment `use_cache` is only supported for `decoder_input_ids` of length 1'
            ngram_hidden_states = [(ngram_embeddings[ngram - 1] + predicting_stream_pos_embed).repeat(batch_size, 1, 1) for ngram in range(self.ngram)]
            extended_attention_mask = None
            extended_predict_attention_mask = None
        else:
            ngram_hidden_states = [ngram_embeddings[ngram - 1] + predicting_stream_pos_embed for ngram in range(self.ngram)]
            extended_attention_mask = self.prepare_attention_mask(hidden_states, attention_mask)
            extended_predict_attention_mask = self.prepare_predict_attention_mask(hidden_states, attention_mask)
        if encoder_attention_mask is not None:
            extended_encoder_attention_mask = (1.0 - encoder_attention_mask[:, None, None, :].repeat(1, self.config.num_decoder_attention_heads, 1, 1)) * torch.finfo(self.dtype).min
            extended_encoder_attention_mask = extended_encoder_attention_mask.to(inputs_embeds.dtype)
        else:
            extended_encoder_attention_mask = None
        hidden_states = torch.cat([hidden_states] + ngram_hidden_states, 1)
        if self.embeddings_layer_norm:
            hidden_states = self.embeddings_layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        all_main_stream_hidden_states = () if output_hidden_states else None
        all_ngram_stream_hidden_states = () if output_hidden_states and self.config.ngram > 0 else None
        all_main_stream_attns = () if output_attentions else None
        all_ngram_stream_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions and self.config.add_cross_attention else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        present_key_values = () if use_cache else None
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                assert attn_mask.size()[0] == len(self.layers), f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_main_stream_hidden_states += (hidden_states[:, :sequence_length],)
                if self.config.ngram > 0:
                    all_ngram_stream_hidden_states += (hidden_states[:, sequence_length:],)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, extended_attention_mask, encoder_hidden_states, extended_encoder_attention_mask, head_mask[idx] if head_mask is not None else None, cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, extended_predict_attention_mask, main_relative_position_buckets, predict_relative_position_buckets, position_ids, None, use_cache, output_attentions)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=extended_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attn_mask=extended_encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, extended_predict_attention_mask=extended_predict_attention_mask, main_relative_position_buckets=main_relative_position_buckets, predict_relative_position_buckets=predict_relative_position_buckets, position_ids=position_ids, past_key_value=past_key_value, use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if use_cache:
                present_key_values += (layer_outputs[4 if output_attentions else 1],)
            if output_attentions:
                all_main_stream_attns += (layer_outputs[1],)
                all_ngram_stream_attns += (layer_outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attns += (layer_outputs[3],)
        if output_hidden_states:
            all_main_stream_hidden_states += (hidden_states[:, :sequence_length],)
            if self.config.ngram > 0:
                all_ngram_stream_hidden_states += (hidden_states[:, sequence_length:],)
        last_hidden_state = hidden_states[:, :sequence_length]
        last_hidden_state_ngram = hidden_states[:, sequence_length:] if self.config.ngram > 0 else None
        if not return_dict:
            return tuple((v for v in [last_hidden_state, last_hidden_state_ngram, present_key_values, all_main_stream_hidden_states, all_ngram_stream_hidden_states, all_main_stream_attns, all_ngram_stream_attns, all_cross_attns] if v is not None))
        return XLMProphetNetDecoderModelOutput(last_hidden_state=last_hidden_state, last_hidden_state_ngram=last_hidden_state_ngram, past_key_values=present_key_values, hidden_states=all_main_stream_hidden_states, hidden_states_ngram=all_ngram_stream_hidden_states, attentions=all_main_stream_attns, ngram_attentions=all_ngram_stream_attns, cross_attentions=all_cross_attns)

    def compute_buffered_relative_buckets(self, position_ids):
        batch_size, sequence_length = position_ids.shape
        position_ids = torch.arange(1, self.max_target_positions).to(position_ids.device).repeat(1, 1)
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(self.num_buckets, self.relative_max_distance, position_ids)
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(batch_size, 1, 1)
        predict_relative_buckets = torch.cat([predict_relative_buckets[:, :sequence_length, :sequence_length], predict_relative_buckets[:, :sequence_length, self.max_target_positions:self.max_target_positions + sequence_length]], 2).repeat(batch_size, 1, 1)
        return (main_relative_buckets, predict_relative_buckets)

    def prepare_attention_mask(self, hidden_states, attention_mask):
        batch_size, seq_length = hidden_states.shape[:2]
        causal_mask = torch.full((seq_length, seq_length), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask = torch.triu(causal_mask, 1)
        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, None, :, :].expand((batch_size, self.config.num_decoder_attention_heads) + causal_mask.shape)
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(self.dtype).min
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask
        return extended_attention_mask.to(hidden_states.dtype)

    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        batch_size, seq_length = hidden_states.shape[:2]
        predict_causal_mask = ngram_attention_bias(self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype)
        predict_causal_mask = torch.cat([predict_causal_mask[:, :seq_length, :seq_length], predict_causal_mask[:, :seq_length, self.max_target_positions:self.max_target_positions + seq_length]], dim=-1)
        extended_predict_causal_mask = predict_causal_mask[None, None, :, :, :].expand((batch_size, self.config.num_decoder_attention_heads) + predict_causal_mask.shape)
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, None, :]) * torch.finfo(self.dtype).min
            extended_attention_mask = extended_attention_mask.expand((batch_size, self.config.num_decoder_attention_heads, self.ngram, seq_length, seq_length))
            extended_attention_mask = torch.cat([extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1)
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask
        return extended_predict_attention_mask.to(hidden_states.dtype)