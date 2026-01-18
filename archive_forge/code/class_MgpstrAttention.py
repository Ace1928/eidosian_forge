import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mgp_str import MgpstrConfig
class MgpstrAttention(nn.Module):

    def __init__(self, config: MgpstrConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attn_drop_rate)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_drop = nn.Dropout(config.drop_rate)

    def forward(self, hidden_states):
        batch_size, num, channel = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(batch_size, num, 3, self.num_heads, channel // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = (qkv[0], qkv[1], qkv[2])
        attention_probs = query @ key.transpose(-2, -1) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)
        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, num, channel)
        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)
        return (context_layer, attention_probs)