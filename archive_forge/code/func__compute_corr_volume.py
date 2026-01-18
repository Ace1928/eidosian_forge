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
def _compute_corr_volume(self, fmap1, fmap2):
    batch_size, num_channels, h, w = fmap1.shape
    fmap1 = fmap1.view(batch_size, num_channels, h * w)
    fmap2 = fmap2.view(batch_size, num_channels, h * w)
    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch_size, h, w, 1, h, w)
    return corr / torch.sqrt(torch.tensor(num_channels))