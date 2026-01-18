import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
class Conv2DDownsample(nn.Module):
    """Downsamples 4x by applying a 2D convolution and doing max pooling."""

    def __init__(self, num_layers: int=1, in_channels: int=3, out_channels: int=64, use_batchnorm: bool=True):
        """
        Constructs a Conv2DDownsample model.

        Args:
          in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
          out_channels (`int`, *optional*, defaults to 64):
            The number of conv output channels.
          use_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether to use batchnorm.
        """
        super().__init__()
        self.conv = Conv2dSamePadding(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv(inputs)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        return out