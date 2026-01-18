import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_swiftformer import SwiftFormerConfig
class SwiftFormerMlp(nn.Module):
    """
    MLP layer with 1*1 convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, in_features: int):
        super().__init__()
        hidden_features = int(in_features * config.mlp_ratio)
        self.norm1 = nn.BatchNorm2d(in_features, eps=config.batch_norm_eps)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        act_layer = ACT2CLS[config.hidden_act]
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(p=0.0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x