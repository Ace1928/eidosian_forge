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
class DPTDepthEstimationHead(nn.Module):
    """
    Output head head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = None
        if config.add_projection:
            self.projection = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        features = config.fusion_hidden_size
        self.head = nn.Sequential(nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.ReLU())

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states[self.config.head_in_index]
        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)
        predicted_depth = self.head(hidden_states)
        predicted_depth = predicted_depth.squeeze(dim=1)
        return predicted_depth