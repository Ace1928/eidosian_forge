import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
class CharactersToMolecules(nn.Module):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=config.downsampling_rate, stride=config.downsampling_rate)
        self.activation = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor:
        cls_encoding = char_encoding[:, 0:1, :]
        char_encoding = torch.transpose(char_encoding, 1, 2)
        downsampled = self.conv(char_encoding)
        downsampled = torch.transpose(downsampled, 1, 2)
        downsampled = self.activation(downsampled)
        downsampled_truncated = downsampled[:, 0:-1, :]
        result = torch.cat([cls_encoding, downsampled_truncated], dim=1)
        result = self.LayerNorm(result)
        return result