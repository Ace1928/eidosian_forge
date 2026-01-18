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
class MaskRCNNPredictor(nn.Sequential):

    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(OrderedDict([('conv5_mask', nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)), ('relu', nn.ReLU(inplace=True)), ('mask_fcn_logits', nn.Conv2d(dim_reduced, num_classes, 1, 1, 0))]))
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')