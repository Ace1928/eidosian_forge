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
class MgpstrLayer(nn.Module):

    def __init__(self, config: MgpstrConfig, drop_path=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MgpstrAttention(config)
        self.drop_path = MgpstrDropPath(drop_path) if drop_path is not None else nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        self.mlp = MgpstrMlp(config, mlp_hidden_dim)

    def forward(self, hidden_states):
        self_attention_outputs = self.attn(self.norm1(hidden_states))
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1]
        hidden_states = self.drop_path(attention_output) + hidden_states
        layer_output = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
        outputs = (layer_output, outputs)
        return outputs