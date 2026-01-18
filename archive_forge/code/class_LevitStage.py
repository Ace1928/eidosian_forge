import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
class LevitStage(nn.Module):
    """
    LeViT Stage consisting of `LevitMLPLayer` and `LevitAttention` layers.
    """

    def __init__(self, config, idx, hidden_sizes, key_dim, depths, num_attention_heads, attention_ratio, mlp_ratio, down_ops, resolution_in):
        super().__init__()
        self.layers = []
        self.config = config
        self.resolution_in = resolution_in
        for _ in range(depths):
            self.layers.append(LevitResidualLayer(LevitAttention(hidden_sizes, key_dim, num_attention_heads, attention_ratio, resolution_in), self.config.drop_path_rate))
            if mlp_ratio > 0:
                hidden_dim = hidden_sizes * mlp_ratio
                self.layers.append(LevitResidualLayer(LevitMLPLayer(hidden_sizes, hidden_dim), self.config.drop_path_rate))
        if down_ops[0] == 'Subsample':
            self.resolution_out = (self.resolution_in - 1) // down_ops[5] + 1
            self.layers.append(LevitAttentionSubsample(*self.config.hidden_sizes[idx:idx + 2], key_dim=down_ops[1], num_attention_heads=down_ops[2], attention_ratio=down_ops[3], stride=down_ops[5], resolution_in=resolution_in, resolution_out=self.resolution_out))
            self.resolution_in = self.resolution_out
            if down_ops[4] > 0:
                hidden_dim = self.config.hidden_sizes[idx + 1] * down_ops[4]
                self.layers.append(LevitResidualLayer(LevitMLPLayer(self.config.hidden_sizes[idx + 1], hidden_dim), self.config.drop_path_rate))
        self.layers = nn.ModuleList(self.layers)

    def get_resolution(self):
        return self.resolution_in

    def forward(self, hidden_state):
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state