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
class VitsWaveNet(torch.nn.Module):

    def __init__(self, config: VitsConfig, num_layers: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(config.wavenet_dropout)
        if hasattr(nn.utils.parametrizations, 'weight_norm'):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm
        if config.speaker_embedding_size != 0:
            cond_layer = torch.nn.Conv1d(config.speaker_embedding_size, 2 * config.hidden_size * num_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name='weight')
        for i in range(num_layers):
            dilation = config.wavenet_dilation_rate ** i
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            in_layer = torch.nn.Conv1d(in_channels=config.hidden_size, out_channels=2 * config.hidden_size, kernel_size=config.wavenet_kernel_size, dilation=dilation, padding=padding)
            in_layer = weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if i < num_layers - 1:
                res_skip_channels = 2 * config.hidden_size
            else:
                res_skip_channels = config.hidden_size
            res_skip_layer = torch.nn.Conv1d(config.hidden_size, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        outputs = torch.zeros_like(inputs)
        num_channels_tensor = torch.IntTensor([self.hidden_size])
        if global_conditioning is not None:
            global_conditioning = self.cond_layer(global_conditioning)
        for i in range(self.num_layers):
            hidden_states = self.in_layers[i](inputs)
            if global_conditioning is not None:
                cond_offset = i * 2 * self.hidden_size
                global_states = global_conditioning[:, cond_offset:cond_offset + 2 * self.hidden_size, :]
            else:
                global_states = torch.zeros_like(hidden_states)
            acts = fused_add_tanh_sigmoid_multiply(hidden_states, global_states, num_channels_tensor[0])
            acts = self.dropout(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_size, :]
                inputs = (inputs + res_acts) * padding_mask
                outputs = outputs + res_skip_acts[:, self.hidden_size:, :]
            else:
                outputs = outputs + res_skip_acts
        return outputs * padding_mask

    def remove_weight_norm(self):
        if self.speaker_embedding_size != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)