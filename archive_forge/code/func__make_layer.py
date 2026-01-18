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
def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], conv_builder: Type[Union[Conv3DSimple, Conv3DNoTemporal, Conv2Plus1D]], planes: int, blocks: int, stride: int=1) -> nn.Sequential:
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        ds_stride = conv_builder.get_downsample_stride(stride)
        downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
    layers = []
    layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(self.inplanes, planes, conv_builder))
    return nn.Sequential(*layers)