import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...ops import boxes as box_ops
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..vgg import VGG, vgg16, VGG16_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
def _vgg_extractor(backbone: VGG, highres: bool, trainable_layers: int):
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if isinstance(b, nn.MaxPool2d)][:-1]
    num_stages = len(stage_indices)
    torch._assert(0 <= trainable_layers <= num_stages, f'trainable_layers should be in the range [0, {num_stages}]. Instead got {trainable_layers}')
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]
    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)
    return SSDFeatureExtractorVGG(backbone, highres)