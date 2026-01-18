from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_resnet import ResNetConfig
class ResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, activation: str='relu'):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut = ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        self.layer = nn.Sequential(ResNetConvLayer(in_channels, out_channels, stride=stride), ResNetConvLayer(out_channels, out_channels, activation=None))
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state