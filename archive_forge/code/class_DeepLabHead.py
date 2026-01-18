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
class DeepLabHead(nn.Sequential):

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(ASPP(in_channels, [12, 24, 36]), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, num_classes, 1))