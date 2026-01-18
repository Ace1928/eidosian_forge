import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
class Wav2Vec2ConformerSelfAttention(nn.Module):
    """Construct an Wav2Vec2ConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.position_embeddings_type = config.position_embeddings_type
        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.attention_dropout)
        if self.position_embeddings_type == 'relative':
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, relative_position_embeddings: Optional[torch.Tensor]=None, output_attentions: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, sequence_length, hidden_size = hidden_states.size()
        query_key_states = hidden_states
        value_states = hidden_states
        if self.position_embeddings_type == 'rotary':
            if relative_position_embeddings is None:
                raise ValueError("`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'")
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.position_embeddings_type == 'relative':
            if relative_position_embeddings is None:
                raise ValueError("`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'relative'")
            scores = self._apply_relative_embeddings(query=query, key=key, relative_position_embeddings=relative_position_embeddings)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)
        return (hidden_states, probs)

    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)
        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., :self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2:]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = hidden_states * cos + rotated_states * sin
        hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)
        return hidden_states

    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(relative_position_embeddings.size(0), -1, self.num_heads, self.head_size)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)
        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, :scores_bd.size(-1) // 2 + 1]
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)
        return scores