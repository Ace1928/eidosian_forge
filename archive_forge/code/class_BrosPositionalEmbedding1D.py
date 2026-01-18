import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_bros import BrosConfig
class BrosPositionalEmbedding1D(nn.Module):

    def __init__(self, config):
        super(BrosPositionalEmbedding1D, self).__init__()
        self.dim_bbox_sinusoid_emb_1d = config.dim_bbox_sinusoid_emb_1d
        inv_freq = 1 / 10000 ** (torch.arange(0.0, self.dim_bbox_sinusoid_emb_1d, 2.0) / self.dim_bbox_sinusoid_emb_1d)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        seq_size = pos_seq.size()
        b1, b2, b3 = seq_size
        sinusoid_inp = pos_seq.view(b1, b2, b3, 1) * self.inv_freq.view(1, 1, 1, self.dim_bbox_sinusoid_emb_1d // 2)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb