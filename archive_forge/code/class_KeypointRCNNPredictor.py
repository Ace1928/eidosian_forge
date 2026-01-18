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
class KeypointRCNNPredictor(nn.Module):

    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(input_features, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1)
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return torch.nn.functional.interpolate(x, scale_factor=float(self.up_scale), mode='bilinear', align_corners=False, recompute_scale_factor=False)