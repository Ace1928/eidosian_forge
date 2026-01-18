import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_deformable_detr import DeformableDetrConfig
class DeformableDetrLoss(nn.Module):
    """
    This class computes the losses for `DeformableDetrForObjectDetection`. The process happens in two steps: 1) we
    compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of
    matched ground-truth / prediction (supervise class and box).

    Args:
        matcher (`DeformableDetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, focal_alpha, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.losses = losses

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if 'logits' not in outputs:
            raise KeyError('No logits were found in the outputs')
        source_logits = outputs['logits']
        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t['class_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1], dtype=source_logits.dtype, layout=source_logits.layout, device=source_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * source_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs['logits']
        device = logits.device
        target_lengths = torch.as_tensor([len(v['class_labels']) for v in targets], device=device)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if 'pred_boxes' not in outputs:
            raise KeyError('No predicted boxes found in outputs')
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_source_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for source, _ in indices])
        return (batch_idx, source_idx)

    def _get_target_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for _, target in indices])
        return (batch_idx, target_idx)

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {'labels': self.loss_labels, 'cardinality': self.loss_cardinality, 'boxes': self.loss_boxes}
        if loss not in loss_map:
            raise ValueError(f'Loss {loss} not supported')
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'auxiliary_outputs' and k != 'enc_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum((len(t['class_labels']) for t in targets))
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'auxiliary_outputs' in outputs:
            for i, auxiliary_outputs in enumerate(outputs['auxiliary_outputs']):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['class_labels'] = torch.zeros_like(bt['class_labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                l_dict = {k + '_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses