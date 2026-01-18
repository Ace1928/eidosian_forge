import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
class SeamlessM4Tv2ConformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        self.layers = nn.ModuleList([SeamlessM4Tv2ConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def _apply_chunk_attention(self, attention_mask, hidden_states):
        """
        Creates a chunk attention mask. It creates a mask to prevent attention across chunks, ensuring that each
        position attends only to positions within its own chunk. If a left chunk overlap is specified
        (`speech_encoder_chunk_size` in the configuration), the attention mask is adjusted accordingly to allow each
        position to also attends the `speech_encoder_chunk_size - 1` previous chunks.
        """
        sequence_len = hidden_states.shape[1]
        chunk_indices = torch.arange(sequence_len, device=hidden_states.device)
        chunk_indices = torch.div(chunk_indices, self.config.speech_encoder_chunk_size).long()
        start_indices = torch.full_like(chunk_indices, 0)
        if self.config.speech_encoder_left_chunk_num >= 0:
            start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp_(min=0)
            start_indices = start_indices * self.config.speech_encoder_chunk_size
            start_indices = start_indices
        start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)
        end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp_(max=sequence_len)
        end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)
        indices = torch.arange(sequence_len, device=hidden_states.device).unsqueeze(0).expand(sequence_len, -1)
        chunk_mask = (indices < start_indices) | (indices >= end_indices)
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = chunk_mask if attention_mask is None else attention_mask.bool() | chunk_mask
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return attention_mask

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        conv_attention_mask = attention_mask
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask.expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
        if self.config.speech_encoder_chunk_size is not None:
            attention_mask = self._apply_chunk_attention(attention_mask, hidden_states)
        if attention_mask is not None:
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        hidden_states = self.dropout(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and dropout_probability < self.config.speech_encoder_layerdrop else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(layer.__call__, hidden_states, attention_mask)
                else:
                    layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, conv_attention_mask=conv_attention_mask)
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)