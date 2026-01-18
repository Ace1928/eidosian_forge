import math
from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_outputs import (
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_van import VanConfig
class VanOverlappingPatchEmbedder(nn.Module):
    """
    Downsamples the input using a patchify operation with a `stride` of 4 by default making adjacent windows overlap by
    half of the area. From [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int=7, stride: int=4):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.normalization = nn.BatchNorm2d(hidden_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state