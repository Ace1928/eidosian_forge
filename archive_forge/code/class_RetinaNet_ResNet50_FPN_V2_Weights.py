import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ...ops import boxes as box_ops, misc as misc_nn_ops, sigmoid_focal_loss
from ...ops.feature_pyramid_network import LastLevelP6P7
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from . import _utils as det_utils
from ._utils import _box_loss, overwrite_eps
from .anchor_utils import AnchorGenerator
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
class RetinaNet_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1 = Weights(url='https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth', transforms=ObjectDetection, meta={**_COMMON_META, 'num_params': 38198935, 'recipe': 'https://github.com/pytorch/vision/pull/5756', '_metrics': {'COCO-val2017': {'box_map': 41.5}}, '_ops': 152.238, '_file_size': 146.037, '_docs': 'These weights were produced using an enhanced training recipe to boost the model accuracy.'})
    DEFAULT = COCO_V1