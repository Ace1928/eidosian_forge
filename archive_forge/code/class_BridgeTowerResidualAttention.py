import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
class BridgeTowerResidualAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.hidden_size, config.hidden_size // 64)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.ModuleDict(OrderedDict([('c_fc', nn.Linear(config.hidden_size, config.hidden_size * 4)), ('gelu', QuickGELUActivation()), ('c_proj', nn.Linear(config.hidden_size * 4, config.hidden_size))]))
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn_mask = None

    def attention(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_state.device)
        self.attn_mask = self.attn_mask.to(dtype=hidden_state.dtype, device=hidden_state.device) if self.attn_mask is not None else None
        return self.attn(hidden_state, hidden_state, hidden_state, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=attention_mask)[0]

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor=None):
        residual_state = hidden_state + self.attention(self.ln_1(hidden_state), attention_mask)
        hidden_state = self.ln_2(residual_state)
        for _, layer in self.mlp.items():
            hidden_state = layer(hidden_state)
        hidden_state = residual_state + hidden_state
        return hidden_state