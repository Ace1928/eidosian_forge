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
class KeypointRCNNHeads(nn.Sequential):

    def __init__(self, in_channels, layers):
        d = []
        next_feature = in_channels
        for out_channels in layers:
            d.append(nn.Conv2d(next_feature, out_channels, 3, stride=1, padding=1))
            d.append(nn.ReLU(inplace=True))
            next_feature = out_channels
        super().__init__(*d)
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)