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
class ConvProjection(nn.Module):
    """
    Project representations from hidden_size*2 back to hidden_size across a window of w = config.upsampling_kernel_size
    characters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv = nn.Conv1d(in_channels=config.hidden_size * 2, out_channels=config.hidden_size, kernel_size=config.upsampling_kernel_size, stride=1)
        self.activation = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs: torch.Tensor, final_seq_char_positions: Optional[torch.Tensor]=None) -> torch.Tensor:
        inputs = torch.transpose(inputs, 1, 2)
        pad_total = self.config.upsampling_kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        pad = nn.ConstantPad1d((pad_beg, pad_end), 0)
        result = self.conv(pad(inputs))
        result = torch.transpose(result, 1, 2)
        result = self.activation(result)
        result = self.LayerNorm(result)
        result = self.dropout(result)
        final_char_seq = result
        if final_seq_char_positions is not None:
            raise NotImplementedError('CanineForMaskedLM is currently not supported')
        else:
            query_seq = final_char_seq
        return query_seq