from functools import partial
from typing import Any, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from ..resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
from ._utils import _SimpleSegmentationModel
from .fcn import FCNHead
class ASPPConv(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]
        super().__init__(*modules)