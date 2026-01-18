import collections
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_bit import BitConfig
class BitDownsampleConv(nn.Module):

    def __init__(self, config, in_channels, out_channels, stride=1, preact=True):
        super().__init__()
        self.conv = WeightStandardizedConv2d(in_channels, out_channels, 1, stride=stride, eps=1e-08, padding=config.global_padding)
        self.norm = nn.Identity() if preact else BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)

    def forward(self, x):
        return self.norm(self.conv(x))