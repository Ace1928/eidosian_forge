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
def _deeplabv3_resnet(backbone: ResNet, num_classes: int, aux: Optional[bool]) -> DeepLabV3:
    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)