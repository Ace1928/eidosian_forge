import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
from ...ops.misc import Conv2dNormActivation
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .. import mobilenet
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .ssd import SSD, SSDScoringHead
def _normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)