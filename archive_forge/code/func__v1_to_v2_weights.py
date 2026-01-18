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
def _v1_to_v2_weights(state_dict, prefix):
    for i in range(4):
        for type in ['weight', 'bias']:
            old_key = f'{prefix}conv.{2 * i}.{type}'
            new_key = f'{prefix}conv.{i}.0.{type}'
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)