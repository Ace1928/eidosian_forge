import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...ops import boxes as box_ops
from ...transforms._presets import ObjectDetection
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _COCO_CATEGORIES
from .._utils import _ovewrite_value_param, handle_legacy_interface
from ..vgg import VGG, vgg16, VGG16_Weights
from . import _utils as det_utils
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
class SSD(nn.Module):
    """
    Implements SSD architecture from `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes, but they will be resized
    to a fixed size before passing it to the backbone.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute with the list of the output channels of
            each feature map. The backbone should return a single Tensor or an OrderedDict[Tensor].
        anchor_generator (DefaultBoxGenerator): module that generates the default boxes for a
            set of feature maps.
        size (Tuple[int, int]): the width and height to which images will be rescaled before feeding them
            to the backbone.
        num_classes (int): number of output classes of the model (including the background).
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        head (nn.Module, optional): Module run on top of the backbone features. Defaults to a module containing
            a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        topk_candidates (int): Number of best detections to keep before NMS.
        positive_fraction (float): a number between 0 and 1 which indicates the proportion of positive
            proposals used during the training of the classification head. It is used to estimate the negative to
            positive ratio.
    """
    __annotations__ = {'box_coder': det_utils.BoxCoder, 'proposal_matcher': det_utils.Matcher}

    def __init__(self, backbone: nn.Module, anchor_generator: DefaultBoxGenerator, size: Tuple[int, int], num_classes: int, image_mean: Optional[List[float]]=None, image_std: Optional[List[float]]=None, head: Optional[nn.Module]=None, score_thresh: float=0.01, nms_thresh: float=0.45, detections_per_img: int=200, iou_thresh: float=0.5, topk_candidates: int=400, positive_fraction: float=0.25, **kwargs: Any):
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        if head is None:
            if hasattr(backbone, 'out_channels'):
                out_channels = backbone.out_channels
            else:
                out_channels = det_utils.retrieve_out_channels(backbone, size)
            if len(out_channels) != len(anchor_generator.aspect_ratios):
                raise ValueError(f'The length of the output channels from the backbone ({len(out_channels)}) do not match the length of the anchor generator aspect ratios ({len(anchor_generator.aspect_ratios)})')
            num_anchors = self.anchor_generator.num_anchors_per_location()
            head = SSDHead(out_channels, num_anchors, num_classes)
        self.head = head
        self.proposal_matcher = det_utils.SSDMatcher(iou_thresh)
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min(size), max(size), image_mean, image_std, size_divisible=1, fixed_size=size, **kwargs)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses
        return detections

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        bbox_regression = head_outputs['bbox_regression']
        cls_logits = head_outputs['cls_logits']
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image, matched_idxs_per_image in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()
            matched_gt_boxes_per_image = targets_per_image['boxes'][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction='sum'))
            gt_classes_target = torch.zeros((cls_logits_per_image.size(0),), dtype=targets_per_image['labels'].dtype, device=targets_per_image['labels'].device)
            gt_classes_target[foreground_idxs_per_image] = targets_per_image['labels'][foreground_matched_idxs_per_image]
            cls_targets.append(gt_classes_target)
        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction='none').view(cls_targets.size())
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float('inf')
        values, idx = negative_loss.sort(1, descending=True)
        background_idxs = idx.sort(1)[1] < num_negative
        N = max(1, num_foreground)
        return {'bbox_regression': bbox_loss.sum() / N, 'classification': (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N}

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]]=None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            if targets is None:
                torch._assert(False, 'targets should not be none when in training mode')
            else:
                for target in targets:
                    boxes = target['boxes']
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(len(boxes.shape) == 2 and boxes.shape[-1] == 4, f'Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.')
                    else:
                        torch._assert(False, f'Expected target boxes to be of type Tensor, got {type(boxes)}.')
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(len(val) == 2, f'expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}')
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target['boxes']
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(False, f'All bounding boxes should have positive height and width. Found invalid box {degen_bb} for target at index {target_idx}.')
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        features = list(features.values())
        head_outputs = self.head(features)
        anchors = self.anchor_generator(images, features)
        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            matched_idxs = []
            if targets is None:
                torch._assert(False, 'targets should not be none when in training mode')
            else:
                for anchors_per_image, targets_per_image in zip(anchors, targets):
                    if targets_per_image['boxes'].numel() == 0:
                        matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device))
                        continue
                    match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
                    matched_idxs.append(self.proposal_matcher(match_quality_matrix))
                losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn('SSD always returns a (Losses, Detections) tuple in scripting')
                self._has_warned = True
            return (losses, detections)
        return self.eager_outputs(losses, detections)

    def postprocess_detections(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs['bbox_regression']
        pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)
        num_classes = pred_scores.size(-1)
        device = pred_scores.device
        detections: List[Dict[str, Tensor]] = []
        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]
                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]
                num_topk = det_utils._topk_min(score, self.topk_candidates, 0)
                score, idxs = score.topk(num_topk)
                box = box[idxs]
                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))
            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            detections.append({'boxes': image_boxes[keep], 'scores': image_scores[keep], 'labels': image_labels[keep]})
        return detections