import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
class Upsample(nn.Module):
    """Upsample module.

    Args:
        scale (`int`):
            Scale factor. Supported scales: 2^n and 3.
        num_features (`int`):
            Channel number of intermediate features.
    """

    def __init__(self, scale, num_features):
        super().__init__()
        self.scale = scale
        if scale & scale - 1 == 0:
            for i in range(int(math.log(scale, 2))):
                self.add_module(f'convolution_{i}', nn.Conv2d(num_features, 4 * num_features, 3, 1, 1))
                self.add_module(f'pixelshuffle_{i}', nn.PixelShuffle(2))
        elif scale == 3:
            self.convolution = nn.Conv2d(num_features, 9 * num_features, 3, 1, 1)
            self.pixelshuffle = nn.PixelShuffle(3)
        else:
            raise ValueError(f'Scale {scale} is not supported. Supported scales: 2^n and 3.')

    def forward(self, hidden_state):
        if self.scale & self.scale - 1 == 0:
            for i in range(int(math.log(self.scale, 2))):
                hidden_state = self.__getattr__(f'convolution_{i}')(hidden_state)
                hidden_state = self.__getattr__(f'pixelshuffle_{i}')(hidden_state)
        elif self.scale == 3:
            hidden_state = self.convolution(hidden_state)
            hidden_state = self.pixelshuffle(hidden_state)
        return hidden_state