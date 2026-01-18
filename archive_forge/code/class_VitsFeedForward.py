import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
class VitsFeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv_1 = nn.Conv1d(config.hidden_size, config.ffn_dim, config.ffn_kernel_size)
        self.conv_2 = nn.Conv1d(config.ffn_dim, config.hidden_size, config.ffn_kernel_size)
        self.dropout = nn.Dropout(config.activation_dropout)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)
        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)
        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)
        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask
        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states