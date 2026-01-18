import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerHungarianMatcher(nn.Module):

    def __init__(self, cost_class: float=1.0, cost_mask: float=1.0, cost_dice: float=1.0, num_points: int=12544):
        """This class computes an assignment between the labels and the predictions of the network.

        For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
        predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
        un-matched (and thus treated as non-objects).

        Params:
            cost_class (float, *optional*, defaults to 1.0):
                This is the relative weight of the classification error in the matching cost.
            cost_mask (float, *optional*,  defaults to 1.0):
                This is the relative weight of the sigmoid ce loss of the binary mask in the matching cost.
            cost_dice (float, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost
            num_points (int, *optional*, defaults to 12544):
                Number of points to be sampled for dice and mask loss matching cost.
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and (cost_dice == 0):
            raise ValueError('All costs cant be 0')
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels) -> List[Tuple[Tensor]]:
        """Performs the matching

        Params:
            masks_queries_logits (`torch.Tensor`):
                A tensor` of dim `batch_size, num_queries, num_labels` with the
                  classification logits.
            class_queries_logits (`torch.Tensor`):
                A tensor` of dim `batch_size, num_queries, height, width` with the
                  predicted masks.

            class_labels (`torch.Tensor`):
                A tensor` of dim `num_target_boxes` (where num_target_boxes is the number
                  of ground-truth objects in the target) containing the class labels.
            mask_labels (`torch.Tensor`):
                A tensor` of dim `num_target_boxes, height, width` containing the target
                  masks.

        Returns:
            `List[Tuple[Tensor]]`: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_targets).
        """
        indices: List[Tuple[np.array]] = []
        num_queries = class_queries_logits.shape[1]
        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits
        for pred_probs, pred_mask, target_mask, labels in zip(preds_probs, preds_masks, mask_labels, class_labels):
            pred_probs = pred_probs.softmax(-1)
            cost_class = -pred_probs[:, labels]
            pred_mask = pred_mask[:, None]
            target_mask = target_mask[:, None].to(pred_mask.device)
            point_coords = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_mask = sample_point(target_mask, point_coords.repeat(target_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            pred_mask = sample_point(pred_mask, point_coords.repeat(pred_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            with autocast(enabled=False):
                pred_mask = pred_mask.float()
                target_mask = target_mask.float()
                cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
                cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
                cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
                cost_matrix = cost_matrix.reshape(num_queries, -1).cpu()
                assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
                indices.append(assigned_indices)
        matched_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return matched_indices