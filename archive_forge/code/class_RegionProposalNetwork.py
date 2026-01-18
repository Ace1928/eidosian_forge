from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation
from . import _utils as det_utils
from .anchor_utils import AnchorGenerator  # noqa: 401
from .image_list import ImageList
class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {'box_coder': det_utils.BoxCoder, 'proposal_matcher': det_utils.Matcher, 'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler}

    def __init__(self, anchor_generator: AnchorGenerator, head: nn.Module, fg_iou_thresh: float, bg_iou_thresh: float, batch_size_per_image: int, positive_fraction: float, pre_nms_top_n: Dict[str, int], post_nms_top_n: Dict[str, int], nms_thresh: float, score_thresh: float=0.0) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 0.001

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return (labels, matched_gt_boxes)

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals: Tensor, objectness: Tensor, image_shapes: List[Tuple[int, int]], num_anchors_per_level: List[int]) -> Tuple[List[Tensor], List[Tensor]]:
        num_images = proposals.shape[0]
        device = proposals.device
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        objectness_prob = torch.sigmoid(objectness)
        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = (boxes[keep], scores[keep], lvl[keep])
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = (boxes[keep], scores[keep], lvl[keep])
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = (boxes[keep], scores[keep])
            final_boxes.append(boxes)
            final_scores.append(scores)
        return (final_boxes, final_scores)

    def compute_loss(self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        box_loss = F.smooth_l1_loss(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1 / 9, reduction='sum') / sampled_inds.numel()
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])
        return (objectness_loss, box_loss)

    def forward(self, images: ImageList, features: Dict[str, Tensor], targets: Optional[List[Dict[str, Tensor]]]=None) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        losses = {}
        if self.training:
            if targets is None:
                raise ValueError('targets should not be None')
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {'loss_objectness': loss_objectness, 'loss_rpn_box_reg': loss_rpn_box_reg}
        return (boxes, losses)