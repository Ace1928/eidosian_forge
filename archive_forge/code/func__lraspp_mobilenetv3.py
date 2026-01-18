from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from ...transforms._presets import SemanticSegmentation
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _VOC_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface, IntermediateLayerGetter
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
def _lraspp_mobilenetv3(backbone: MobileNetV3, num_classes: int) -> LRASPP:
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, '_is_cn', False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]
    high_pos = stage_indices[-1]
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): 'low', str(high_pos): 'high'})
    return LRASPP(backbone, low_channels, high_channels, num_classes)