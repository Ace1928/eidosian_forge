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
class ConvGRU(nn.Module):
    """Convolutional Gru unit."""

    def __init__(self, *, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.convz = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h