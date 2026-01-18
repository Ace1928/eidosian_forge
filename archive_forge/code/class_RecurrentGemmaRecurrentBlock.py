import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_recurrent_gemma import RecurrentGemmaConfig
class RecurrentGemmaRecurrentBlock(nn.Module):
    """Griffin and Hawk's recurrent block."""

    def __init__(self, config):
        super().__init__()
        self.lru_width = config.lru_width
        self.hidden_size = config.hidden_size
        self.linear_y = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_x = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_out = nn.Linear(in_features=config.lru_width, out_features=config.hidden_size)
        self.conv1d_width = config.conv1d_width
        self.conv_1d = nn.Conv1d(config.lru_width, config.lru_width, kernel_size=config.conv1d_width, groups=config.lru_width, padding=config.conv1d_width - 1)
        self.rg_lru = RecurrentGemmaRglru(config)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.conv1d_state = None

    def forward(self, input_states: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor, cache_position: torch.Tensor, use_cache: bool=True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, seq_len, _ = input_states.shape
        y_branch = self.linear_y(input_states)
        y_branch = self.act_fn(y_branch)
        x_branch = self.linear_x(input_states)
        x_branch = x_branch.transpose(1, 2)
        if use_cache:
            if cache_position.shape[0] != 1:
                self.conv1d_state = nn.functional.pad(x_branch, (self.conv1d_width - x_branch.shape[-1] - 1, 0))
                x_branch = self.conv_1d(x_branch)[..., :seq_len]
            else:
                conv_state = torch.cat((self.conv1d_state, x_branch), -1)
                x_branch = torch.sum(conv_state * self.conv_1d.weight[:, 0, :], dim=-1) + self.conv_1d.bias
                x_branch = x_branch.unsqueeze(-1)
                self.conv1d_state = conv_state[:, :, 1:]
        else:
            x_branch = self.conv_1d(x_branch)[..., :seq_len]
        x_branch = self.rg_lru(x_branch.transpose(1, 2), position_ids)
        hidden_states = x_branch * y_branch
        hidden_states = self.linear_out(hidden_states)
        return hidden_states

    def _setup_cache(self, batch, device, dtype):
        self.rg_lru.recurrent_states = torch.zeros((batch, self.lru_width), device=device, dtype=torch.float32)
        self.conv1d_state = torch.zeros((batch, self.hidden_size, self.conv1d_width - 1), device=device, dtype=dtype)