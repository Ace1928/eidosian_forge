import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaSimpleRelativePositionalBias(nn.Module):
    """
    Simple relative positional embeddings copied from the Mega repo; renamed variables for better readability
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.max_positions = self.config.max_positions if self.config.chunk_size < 0 else self.config.chunk_size
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * config.max_positions - 1))

    def forward(self, seq_len):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))
        bias = self.rel_pos_bias[self.max_positions - seq_len:self.max_positions + seq_len - 1]
        tile = F.pad(bias, (0, seq_len))
        tile = torch.tile(tile, (seq_len,))
        tile = tile[:-seq_len]
        tile = tile.view(seq_len, 3 * seq_len - 2)
        start = (2 * seq_len - 1) // 2
        end = tile.size(1) - start
        tile = tile[:, start:end]
        return tile