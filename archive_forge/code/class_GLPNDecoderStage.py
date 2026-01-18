import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_glpn import GLPNConfig
class GLPNDecoderStage(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        should_skip = in_channels == out_channels
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1) if not should_skip else nn.Identity()
        self.fusion = GLPNSelectiveFeatureFusion(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, hidden_state, residual=None):
        hidden_state = self.convolution(hidden_state)
        if residual is not None:
            hidden_state = self.fusion(hidden_state, residual)
        hidden_state = self.upsample(hidden_state)
        return hidden_state
        hidden_state = self.upsample(hidden_state)
        return hidden_state