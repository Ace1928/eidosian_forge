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
class FCOS(nn.Module):
    """
    Implements FCOS.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps. For FCOS, only set one anchor for per position of each level, the width and height equal to
            the stride of feature map, and set aspect ratio = 1.0, so the center of anchor is equivalent to the point
            in FCOS paper.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FCOS
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # FCOS needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((8,), (16,), (32,), (64,), (128,)),
        >>>     aspect_ratios=((1.0,),)
        >>> )
        >>>
        >>> # put the pieces together inside a FCOS model
        >>> model = FCOS(
        >>>     backbone,
        >>>     num_classes=80,
        >>>     anchor_generator=anchor_generator,
        >>> )
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    __annotations__ = {'box_coder': det_utils.BoxLinearCoder}

    def __init__(self, backbone: nn.Module, num_classes: int, min_size: int=800, max_size: int=1333, image_mean: Optional[List[float]]=None, image_std: Optional[List[float]]=None, anchor_generator: Optional[AnchorGenerator]=None, head: Optional[nn.Module]=None, center_sampling_radius: float=1.5, score_thresh: float=0.2, nms_thresh: float=0.6, detections_per_img: int=100, topk_candidates: int=1000, **kwargs):
        super().__init__()
        _log_api_usage_once(self)
        if not hasattr(backbone, 'out_channels'):
            raise ValueError('backbone should contain an attribute out_channels specifying the number of output channels (assumed to be the same for all the levels)')
        self.backbone = backbone
        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(f'anchor_generator should be of type AnchorGenerator or None, instead  got {type(anchor_generator)}')
        if anchor_generator is None:
            anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
            aspect_ratios = ((1.0,),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator
        if self.anchor_generator.num_anchors_per_location()[0] != 1:
            raise ValueError(f'anchor_generator.num_anchors_per_location()[0] should be 1 instead of {anchor_generator.num_anchors_per_location()[0]}')
        if head is None:
            head = FCOSHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head
        self.box_coder = det_utils.BoxLinearCoder(normalize_by_size=True)
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
        self.center_sampling_radius = center_sampling_radius
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if self.training:
            return losses
        return detections

    def compute_loss(self, targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor], num_anchors_per_level: List[int]) -> Dict[str, Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device))
                continue
            gt_boxes = targets_per_image['boxes']
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
            anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(dim=2).values < self.center_sampling_radius * anchor_sizes[:, None]
            x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)
            x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)
            pairwise_match &= pairwise_dist.min(dim=2).values > 0
            lower_bound = anchor_sizes * 4
            lower_bound[:num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes * 8
            upper_bound[-num_anchors_per_level[-1]:] = float('inf')
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            pairwise_match = pairwise_match.to(torch.float32) * (100000000.0 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)
            matched_idx[min_values < 1e-05] = -1
            matched_idxs.append(matched_idx)
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs: Dict[str, List[Tensor]], anchors: List[List[Tensor]], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        class_logits = head_outputs['cls_logits']
        box_regression = head_outputs['bbox_regression']
        box_ctrness = head_outputs['bbox_ctrness']
        num_images = len(image_shapes)
        detections: List[Dict[str, Tensor]] = []
        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            box_ctrness_per_image = [bc[index] for bc in box_ctrness]
            anchors_per_image, image_shape = (anchors[index], image_shapes[index])
            image_boxes = []
            image_scores = []
            image_labels = []
            for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level in zip(box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image):
                num_classes = logits_per_level.shape[-1]
                scores_per_level = torch.sqrt(torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]
                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode='floor')
                labels_per_level = topk_idxs % num_classes
                boxes_per_level = self.box_coder.decode(box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs])
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)
                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)
            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]
            detections.append({'boxes': image_boxes[keep], 'scores': image_scores[keep], 'labels': image_labels[keep]})
        return detections

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]]=None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:
            if targets is None:
                torch._assert(False, 'targets should not be none when in training mode')
            else:
                for target in targets:
                    boxes = target['boxes']
                    torch._assert(isinstance(boxes, torch.Tensor), 'Expected target boxes to be of type Tensor.')
                    torch._assert(len(boxes.shape) == 2 and boxes.shape[-1] == 4, f'Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.')
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
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, 'targets should not be none when in training mode')
            else:
                losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
        else:
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn('FCOS always returns a (Losses, Detections) tuple in scripting')
                self._has_warned = True
            return (losses, detections)
        return self.eager_outputs(losses, detections)