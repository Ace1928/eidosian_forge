from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from ...ops import misc as misc_nn_ops
from ...transforms._presets import ObjectDetection
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from ..resnet import resnet50, ResNet50_Weights
from ._utils import overwrite_eps
from .anchor_utils import AnchorGenerator
from .backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor, _validate_trainable_layers
from .generalized_rcnn import GeneralizedRCNN
from .roi_heads import RoIHeads
from .rpn import RegionProposalNetwork, RPNHead
from .transform import GeneralizedRCNNTransform
class FasterRCNN_MobileNet_V3_Large_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(url='https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth', transforms=ObjectDetection, meta={**_COMMON_META, 'num_params': 19386354, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-mobilenetv3-large-fpn', '_metrics': {'COCO-val2017': {'box_map': 32.8}}, '_ops': 4.494, '_file_size': 74.239, '_docs': 'These weights were produced by following a similar training recipe as on the paper.'})
    DEFAULT = COCO_V1