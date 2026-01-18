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
class PixelShuffleUpsampler(nn.Module):

    def __init__(self, config, num_features):
        super().__init__()
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.upsample = Upsample(config.upscale, num_features)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output):
        x = self.conv_before_upsample(sequence_output)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.final_convolution(x)
        return x