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
class FCOSHead(nn.Module):
    """
    A regression and classification head for use in FCOS.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer of head. Default: 4.
    """
    __annotations__ = {'box_coder': det_utils.BoxLinearCoder}

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: Optional[int]=4) -> None:
        super().__init__()
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)
        self.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes, num_convs)
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits = head_outputs['cls_logits']
        bbox_regression = head_outputs['bbox_regression']
        bbox_ctrness = head_outputs['bbox_ctrness']
        all_gt_classes_targets = []
        all_gt_boxes_targets = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image['labels']) == 0:
                gt_classes_targets = targets_per_image['labels'].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image['boxes'].new_zeros((len(matched_idxs_per_image), 4))
            else:
                gt_classes_targets = targets_per_image['labels'][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image['boxes'][matched_idxs_per_image.clip(min=0)]
            gt_classes_targets[matched_idxs_per_image < 0] = -1
            all_gt_classes_targets.append(gt_classes_targets)
            all_gt_boxes_targets.append(gt_boxes_targets)
        all_gt_boxes_targets, all_gt_classes_targets, anchors = (torch.stack(all_gt_boxes_targets), torch.stack(all_gt_classes_targets), torch.stack(anchors))
        foregroud_mask = all_gt_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foregroud_mask, all_gt_classes_targets[foregroud_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction='sum')
        pred_boxes = self.box_coder.decode(bbox_regression, anchors)
        loss_bbox_reg = generalized_box_iou_loss(pred_boxes[foregroud_mask], all_gt_boxes_targets[foregroud_mask], reduction='sum')
        bbox_reg_targets = self.box_coder.encode(anchors, all_gt_boxes_targets)
        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0] * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = nn.functional.binary_cross_entropy_with_logits(pred_centerness[foregroud_mask], gt_ctrness_targets[foregroud_mask], reduction='sum')
        return {'classification': loss_cls / max(1, num_foreground), 'bbox_regression': loss_bbox_reg / max(1, num_foreground), 'bbox_ctrness': loss_bbox_ctrness / max(1, num_foreground)}

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        cls_logits = self.classification_head(x)
        bbox_regression, bbox_ctrness = self.regression_head(x)
        return {'cls_logits': cls_logits, 'bbox_regression': bbox_regression, 'bbox_ctrness': bbox_ctrness}