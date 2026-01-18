import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class Inception(nn.Module):

    def __init__(self, in_channels: int, ch1x1: int, ch3x3red: int, ch3x3: int, ch5x5red: int, ch5x5: int, pool_proj: int, conv_block: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(conv_block(in_channels, ch5x5red, kernel_size=1), conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True), conv_block(in_channels, pool_proj, kernel_size=1))

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)