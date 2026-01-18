import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_dinat import DinatConfig
class DinatDownsampler(nn.Module):
    """
    Convolutional Downsampling Layer.

    Args:
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, dim: int, norm_layer: nn.Module=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        input_feature = self.reduction(input_feature.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        input_feature = self.norm(input_feature)
        return input_feature