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
class VitsDilatedDepthSeparableConv(nn.Module):

    def __init__(self, config: VitsConfig, dropout_rate=0.0):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.convs_dilated = nn.ModuleList()
        self.convs_pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_dilated.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, dilation=dilation, padding=padding))
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def forward(self, inputs, padding_mask, global_conditioning=None):
        if global_conditioning is not None:
            inputs = inputs + global_conditioning
        for i in range(self.num_layers):
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.convs_pointwise[i](hidden_states)
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            inputs = inputs + hidden_states
        return inputs * padding_mask