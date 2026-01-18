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
class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(self) -> None:
        super().__init__(nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False), nn.BatchNorm3d(45), nn.ReLU(inplace=True), nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True))