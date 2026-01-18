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
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(list(x.shape[2:]) == [1, 1], f'x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}')
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return (scores, bbox_deltas)