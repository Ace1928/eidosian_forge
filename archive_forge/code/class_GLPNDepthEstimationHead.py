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
class GLPNDepthEstimationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = config.decoder_hidden_size
        self.head = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=False), nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states[self.config.head_in_index]
        hidden_states = self.head(hidden_states)
        predicted_depth = torch.sigmoid(hidden_states) * self.config.max_depth
        predicted_depth = predicted_depth.squeeze(dim=1)
        return predicted_depth