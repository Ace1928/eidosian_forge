from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch.nn as nn
from torch import Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .._utils import _ModelURLs
class Conv2Plus1D(nn.Sequential):

    def __init__(self, in_planes: int, out_planes: int, midplanes: int, stride: int=1, padding: int=1) -> None:
        super().__init__(nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, padding, padding), bias=False), nn.BatchNorm3d(midplanes), nn.ReLU(inplace=True), nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False))

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return (stride, stride, stride)