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
class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)

    Used in lightweight SR to save parameters.

    Args:
        scale (int):
            Scale factor. Supported scales: 2^n and 3.
        in_channels (int):
            Channel number of intermediate features.
        out_channels (int):
            Channel number of output features.
    """

    def __init__(self, scale, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, scale ** 2 * out_channels, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x