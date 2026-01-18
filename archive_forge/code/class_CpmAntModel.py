import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig
@add_start_docstrings('The bare CPMAnt Model outputting raw hidden-states without any specific head on top.', CPMANT_START_DOCSTRING)
class CpmAntModel(CpmAntPreTrainedModel):

    def __init__(self, config: CpmAntConfig):
        super().__init__(config)
        self.encoder = CpmAntEncoder(config)
        self.segment_embedding = nn.Embedding(config.segment_types, config.hidden_size)
        self.input_embedding = nn.Embedding(config.vocab_size + config.prompt_types * config.prompt_length, config.hidden_size)
        self.position_bias = CpmAntSegmentPositionEmbedding(config)
        self.prompt_length = config.prompt_length
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.input_embedding

    def set_input_embeddings(self, embeddings, **kwargs):
        self.input_embedding = embeddings

    def _prepare_attention_mask(self, input_ids, span, context, length):
        batch = input_ids.size(0)
        seqlen = input_ids.size(1)
        device = input_ids.device
        directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
        attention_mask = context[:, None, :] | context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
        attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
        mask_1d = torch.tensor(list(range(seqlen - self.prompt_length))[::-1], device=device)[None, :].repeat(batch, 1) < length[:, None]
        mask_1d = torch.cat((torch.ones(batch, self.prompt_length, device=device).bool(), mask_1d), dim=1)
        attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
        return attention_mask

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, use_cache: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if input_ids.dtype != torch.int32:
            input_ids = input_ids.to(torch.int32)
        dtype, device = (input_ids.dtype, input_ids.device)
        segment = torch.where(input_ids != 0, 2, 0).to(dtype=dtype, device=device)
        length = (segment != 0).sum(-1).to(dtype=dtype, device=device)
        input_ids = torch.cat((torch.arange(self.prompt_length * 2 + self.vocab_size, self.prompt_length * 3 + self.vocab_size, dtype=dtype, device=device).repeat(input_ids.size(0), 1), input_ids), dim=1)
        batch, seq_length = input_ids.size()
        segment = torch.cat((torch.zeros(batch, self.prompt_length, dtype=dtype, device=device), segment), dim=1)
        context = torch.full((batch, seq_length), 1, dtype=dtype, device=device)
        position = torch.arange(seq_length, dtype=dtype, device=device).repeat(batch, 1)
        span = torch.full((batch, seq_length), 0, dtype=dtype, device=device)
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.encoder.num_layers)
            input_ids = input_ids.contiguous()
            hidden_states = self.input_embedding(input_ids)
            segment_states = self.segment_embedding(segment)
            hidden_states = hidden_states + segment_states
        else:
            past_length = past_key_values[0][0].size(-2)
            segment_states = self.segment_embedding(segment)
            hidden_states = self.input_embedding(input_ids) + segment_states[:, -1:, :]
        attention_mask = self._prepare_attention_mask(input_ids, span, context, length)
        position_bias = self.position_bias(position, position, segment, segment)
        attention_mask = attention_mask[:, past_length:, :]
        position_bias = position_bias[:, :, past_length:, :]
        hidden_states = hidden_states[:, past_length:, :]
        hidden_states, present_key_values, all_hidden_states, all_attentions = self.encoder(hidden_states, attention_mask, position_bias, output_attentions, output_hidden_states, past_key_values, use_cache)
        if past_length == 0:
            hidden_states = hidden_states[:, self.prompt_length:, :]
            if all_attentions is not None:
                new_attentions = ()
                for attention in all_attentions:
                    new_attentions += (attention[:, :, self.prompt_length:, self.prompt_length:],)
                all_attentions = new_attentions
            if all_hidden_states is not None:
                new_hidden_states = ()
                for hidden_state in all_hidden_states:
                    new_hidden_states += (hidden_state[:, self.prompt_length:, :],)
                all_hidden_states = new_hidden_states
        if not return_dict:
            return tuple((v for v in [hidden_states, present_key_values, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=present_key_values, hidden_states=all_hidden_states, attentions=all_attentions)