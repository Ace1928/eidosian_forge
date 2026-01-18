from typing import Any, Optional
import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from ...ops import misc as misc_nn_ops
from ...transforms._presets import ObjectDetection
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_PERSON_CATEGORIES, _COCO_PERSON_KEYPOINT_NAMES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from ._utils import overwrite_eps
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .faster_rcnn import FasterRCNN
class KeypointRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_LEGACY = Weights(url='https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth', transforms=ObjectDetection, meta={**_COMMON_META, 'num_params': 59137258, 'recipe': 'https://github.com/pytorch/vision/issues/1606', '_metrics': {'COCO-val2017': {'box_map': 50.6, 'kp_map': 61.1}}, '_ops': 133.924, '_file_size': 226.054, '_docs': '\n                These weights were produced by following a similar training recipe as on the paper but use a checkpoint\n                from an early epoch.\n            '})
    COCO_V1 = Weights(url='https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth', transforms=ObjectDetection, meta={**_COMMON_META, 'num_params': 59137258, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/detection#keypoint-r-cnn', '_metrics': {'COCO-val2017': {'box_map': 54.6, 'kp_map': 65.0}}, '_ops': 137.42, '_file_size': 226.054, '_docs': 'These weights were produced by following a similar training recipe as on the paper.'})
    DEFAULT = COCO_V1