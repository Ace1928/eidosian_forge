import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
class ClvpDecoder(ClvpPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ClvpDecoderLayer`]
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_embeds_layer = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.position_embeds_layer = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.drop = nn.Dropout(self.config.embd_pdrop)
        self.layers = nn.ModuleList([ClvpDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layer = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.layers[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CLVP_DECODER_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            inputs_embeds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if past_key_values is None:
            past_key_values_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_key_values_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_key_values_length, input_shape[-1] + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        if inputs_embeds is None:
            inputs_embeds = self.input_embeds_layer(input_ids)
        position_embeds = self.position_embeds_layer(position_ids)
        inputs_embeds = inputs_embeds + position_embeds
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        hidden_states = inputs_embeds
        if token_type_ids is not None:
            token_type_embeds = self.input_embeds_layer(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = torch.utils.checkpoint.checkpoint(block.__call__, hidden_states, None, attention_mask, position_ids, head_mask[i])
            else:
                outputs = block(hidden_states, past_key_value=past_key_value, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions, cross_attentions=all_cross_attentions)