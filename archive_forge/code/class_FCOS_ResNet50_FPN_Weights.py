import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ...ops import boxes as box_ops, generalized_box_iou_loss, misc as misc_nn_ops, sigmoid_focal_loss
from ...ops.feature_pyramid_network import LastLevelP6P7
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from . import _utils as det_utils
from .anchor_utils import AnchorGenerator
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
class FCOS_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(url='https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth', transforms=ObjectDetection, meta={'num_params': 32269600, 'categories': _COCO_CATEGORIES, 'min_size': (1, 1), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/detection#fcos-resnet-50-fpn', '_metrics': {'COCO-val2017': {'box_map': 39.2}}, '_ops': 128.207, '_file_size': 123.608, '_docs': 'These weights were produced by following a similar training recipe as on the paper.'})
    DEFAULT = COCO_V1