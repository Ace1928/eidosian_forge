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
class SeamlessM4Tv2VariancePredictor(nn.Module):

    def __init__(self, embed_dim, hidden_dim, kernel_size, var_pred_dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.activation_fuction = nn.ReLU()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout_module = nn.Dropout(p=var_pred_dropout)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding='same')
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: Tensor, padding_mask: Tensor=None) -> Tensor:
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv1(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln1(hidden_states))
        if padding_mask is not None:
            hidden_states = hidden_states.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)
        hidden_states = self.conv2(hidden_states.transpose(1, 2))
        hidden_states = self.activation_fuction(hidden_states).transpose(1, 2)
        hidden_states = self.dropout_module(self.ln2(hidden_states))
        return self.proj(hidden_states).squeeze(dim=2)