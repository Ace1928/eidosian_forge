import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_trajectory_transformer import TrajectoryTransformerConfig
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f'n_head ({config.n_head}) should be a divisor of n_embd ({config.n_embd})')
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size), persistent=False)
        joined_dim = config.observation_dim + config.action_dim + 2
        self.mask.squeeze()[:, joined_dim - 1::joined_dim] = 0
        self.n_head = config.n_head

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], layer_past: Optional[Tuple[torch.Tensor]]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False):
        batch_size, sequence_length, embedding_dim = hidden_states.size()
        key = self.key(hidden_states).view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2)
        query = self.query(hidden_states).view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2)
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :sequence_length, :sequence_length] == 0, torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        self._attn_map = attn_weights.clone()
        attn_weights = self.attn_drop(attn_weights)
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        output = self.resid_drop(self.proj(output))
        outputs = (output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs