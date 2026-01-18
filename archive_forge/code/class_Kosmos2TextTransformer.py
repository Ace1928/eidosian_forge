import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_kosmos2 import Kosmos2Config, Kosmos2TextConfig, Kosmos2VisionConfig
class Kosmos2TextTransformer(nn.Module):
    """
    Transformer decoder consisting of `config.layers` layers. Each layer is a [`Kosmos2TextBlock`].

    Args:
        config: Kosmos2TextConfig
    """

    def __init__(self, config: Kosmos2TextConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.embed_scale = math.sqrt(config.embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
        self.embed_positions = Kosmos2TextSinusoidalPositionalEmbedding(num_positions=config.max_position_embeddings, embedding_dim=config.embed_dim, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([Kosmos2TextBlock(config) for _ in range(config.layers)])
        self.layer_norm = nn.LayerNorm(config.embed_dim, config.layer_norm_eps)
        self.gradient_checkpointing = False

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length)
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        return combined_attention_mask

    def forward_embedding(self, input_ids, inputs_embeds: torch.Tensor=None, image_embeds: torch.Tensor=None, img_input_mask: torch.Tensor=None, past_key_values_length: int=0, position_ids: torch.Tensor=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if image_embeds is not None:
            inputs_embeds[img_input_mask.to(dtype=torch.bool)] = image_embeds.to(inputs_embeds.device).view(-1, image_embeds.size(-1))
        inputs_embeds = inputs_embeds * self.embed_scale
        positions = self.embed_positions(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length, position_ids=position_ids)
        positions = positions.to(inputs_embeds.device)
        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states

    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, image_embeds: Optional[torch.Tensor]=None, image_embeds_position_mask: Optional[torch.Tensor]=None, encoder_hidden_states: Optional[torch.Tensor]=None, encoder_attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if past_key_values_length > 0:
            image_embeds = None
            image_embeds_position_mask = None
        hidden_states = self.forward_embedding(input_ids=input_ids, inputs_embeds=inputs_embeds, image_embeds=image_embeds, img_input_mask=image_embeds_position_mask, past_key_values_length=past_key_values_length, position_ids=position_ids)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, hidden_states, past_key_values_length)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if output_attentions and encoder_hidden_states is not None else None
        present_key_value_states = () if use_cache else None
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ['head_mask', 'cross_attn_head_mask']):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(f'The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(decoder_layer.__call__, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask[idx] if head_mask is not None else None, cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, None, output_attentions, use_cache)
            else:
                layer_outputs = decoder_layer(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, cross_attn_layer_head_mask=cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            hidden_states = layer_outputs[0]
            if use_cache:
                present_key_value_states += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, present_key_value_states, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=present_key_value_states, hidden_states=all_hidden_states, attentions=all_self_attns, cross_attentions=all_cross_attentions)