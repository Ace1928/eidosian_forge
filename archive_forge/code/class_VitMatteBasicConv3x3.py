from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig
class VitMatteBasicConv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """

    def __init__(self, config, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        self.relu = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.relu(hidden_state)
        return hidden_state