from collections import OrderedDict
from typing import Any, Callable, Optional
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from ...ops import misc as misc_nn_ops
from ...transforms._presets import ObjectDetection
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..resnet import resnet50, ResNet50_Weights
from ._utils import overwrite_eps
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead
class MaskRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(url='https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth', transforms=ObjectDetection, meta={**_COMMON_META, 'num_params': 44401393, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/detection#mask-r-cnn', '_metrics': {'COCO-val2017': {'box_map': 37.9, 'mask_map': 34.6}}, '_ops': 134.38, '_file_size': 169.84, '_docs': 'These weights were produced by following a similar training recipe as on the paper.'})
    DEFAULT = COCO_V1