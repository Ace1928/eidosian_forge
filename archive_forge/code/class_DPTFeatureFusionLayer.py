import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
class DPTFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
        align_corners (`bool`, *optional*, defaults to `True`):
            The align_corner setting for bilinear upsample.
    """

    def __init__(self, config, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)
        self.residual_layer1 = DPTPreActResidualLayer(config)
        self.residual_layer2 = DPTPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode='bilinear', align_corners=False)
            hidden_state = hidden_state + self.residual_layer1(residual)
        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(hidden_state, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        hidden_state = self.projection(hidden_state)
        return hidden_state