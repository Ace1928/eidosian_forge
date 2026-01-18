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
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
class DetaStage1Assigner(nn.Module):

    def __init__(self, t_low=0.3, t_high=0.7, max_k=4):
        super().__init__()
        self.positive_fraction = 0.5
        self.batch_size_per_image = 256
        self.k = max_k
        self.t_low = t_low
        self.t_high = t_high
        self.anchor_matcher = DetaMatcher(thresholds=[t_low, t_high], labels=[0, -1, 1], allow_low_quality_matches=True)

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite the label vector to the ignore value
        (-1) for all elements that are not included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(label, self.batch_size_per_image, self.positive_fraction, 0)
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    def forward(self, outputs, targets):
        bs = len(targets)
        indices = []
        for b in range(bs):
            anchors = outputs['anchors'][b]
            if len(targets[b]['boxes']) == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=anchors.device), torch.tensor([], dtype=torch.long, device=anchors.device)))
                continue
            iou, _ = box_iou(center_to_corners_format(targets[b]['boxes']), center_to_corners_format(anchors))
            matched_idxs, matched_labels = self.anchor_matcher(iou)
            matched_labels = self._subsample_labels(matched_labels)
            all_pr_inds = torch.arange(len(anchors), device=matched_labels.device)
            pos_pr_inds = all_pr_inds[matched_labels == 1]
            pos_gt_inds = matched_idxs[pos_pr_inds]
            pos_pr_inds, pos_gt_inds = self.postprocess_indices(pos_pr_inds, pos_gt_inds, iou)
            pos_pr_inds, pos_gt_inds = (pos_pr_inds.to(anchors.device), pos_gt_inds.to(anchors.device))
            indices.append((pos_pr_inds, pos_gt_inds))
        return indices

    def postprocess_indices(self, pr_inds, gt_inds, iou):
        return sample_topk_per_gt(pr_inds, gt_inds, iou, self.k)