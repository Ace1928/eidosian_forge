from functools import partial
from typing import Any, Optional
from torch import nn
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
class FCNHead(nn.Sequential):

    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), nn.BatchNorm2d(inter_channels), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(inter_channels, channels, 1)]
        super().__init__(*layers)