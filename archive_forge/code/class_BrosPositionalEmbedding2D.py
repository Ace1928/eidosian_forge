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
class BrosPositionalEmbedding2D(nn.Module):

    def __init__(self, config):
        super(BrosPositionalEmbedding2D, self).__init__()
        self.dim_bbox = config.dim_bbox
        self.x_pos_emb = BrosPositionalEmbedding1D(config)
        self.y_pos_emb = BrosPositionalEmbedding1D(config)

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        stack = []
        for i in range(self.dim_bbox):
            if i % 2 == 0:
                stack.append(self.x_pos_emb(bbox[..., i]))
            else:
                stack.append(self.y_pos_emb(bbox[..., i]))
        bbox_pos_emb = torch.cat(stack, dim=-1)
        return bbox_pos_emb