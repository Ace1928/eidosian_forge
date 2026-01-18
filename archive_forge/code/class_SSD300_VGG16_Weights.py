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
class SSD300_VGG16_Weights(WeightsEnum):
    COCO_V1 = Weights(url='https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth', transforms=ObjectDetection, meta={'num_params': 35641826, 'categories': _COCO_CATEGORIES, 'min_size': (1, 1), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/detection#ssd300-vgg16', '_metrics': {'COCO-val2017': {'box_map': 25.1}}, '_ops': 34.858, '_file_size': 135.988, '_docs': 'These weights were produced by following a similar training recipe as on the paper.'})
    DEFAULT = COCO_V1