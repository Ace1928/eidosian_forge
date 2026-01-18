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
class NearestConvUpsampler(nn.Module):

    def __init__(self, config, num_features):
        super().__init__()
        if config.upscale != 4:
            raise ValueError('The nearest+conv upsampler only supports an upscale factor of 4 at the moment.')
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, sequence_output):
        sequence_output = self.conv_before_upsample(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode='nearest')))
        sequence_output = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode='nearest')))
        reconstruction = self.final_convolution(self.lrelu(self.conv_hr(sequence_output)))
        return reconstruction