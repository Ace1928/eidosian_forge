from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation
from ...transforms._presets import OpticalFlow
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._utils import handle_legacy_interface
from ._utils import grid_sample, make_coords_grid, upsample_flow
class MotionEncoder(nn.Module):
    """The motion encoder, part of the update block.

    Takes the current predicted flow and the correlation features as input and returns an encoded version of these.
    """

    def __init__(self, *, in_channels_corr, corr_layers=(256, 192), flow_layers=(128, 64), out_channels=128):
        super().__init__()
        if len(flow_layers) != 2:
            raise ValueError(f'The expected number of flow_layers is 2, instead got {len(flow_layers)}')
        if len(corr_layers) not in (1, 2):
            raise ValueError(f'The number of corr_layers should be 1 or 2, instead got {len(corr_layers)}')
        self.convcorr1 = Conv2dNormActivation(in_channels_corr, corr_layers[0], norm_layer=None, kernel_size=1)
        if len(corr_layers) == 2:
            self.convcorr2 = Conv2dNormActivation(corr_layers[0], corr_layers[1], norm_layer=None, kernel_size=3)
        else:
            self.convcorr2 = nn.Identity()
        self.convflow1 = Conv2dNormActivation(2, flow_layers[0], norm_layer=None, kernel_size=7)
        self.convflow2 = Conv2dNormActivation(flow_layers[0], flow_layers[1], norm_layer=None, kernel_size=3)
        self.conv = Conv2dNormActivation(corr_layers[-1] + flow_layers[-1], out_channels - 2, norm_layer=None, kernel_size=3)
        self.out_channels = out_channels

    def forward(self, flow, corr_features):
        corr = self.convcorr1(corr_features)
        corr = self.convcorr2(corr)
        flow_orig = flow
        flow = self.convflow1(flow)
        flow = self.convflow2(flow)
        corr_flow = torch.cat([corr, flow], dim=1)
        corr_flow = self.conv(corr_flow)
        return torch.cat([corr_flow, flow_orig], dim=1)