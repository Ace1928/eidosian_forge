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
class DetaMatcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth element. Each predicted element will
    have exactly zero or one matches; each ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes how well each (ground-truth,
    prediction)-pair match each other. For example, if the elements are boxes, this matrix may contain box
    intersection-over-union overlap values.

    The matcher returns (a) a vector of length N containing the index of the ground-truth element m in [0, M) that
    matches to prediction n in [0, N). (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool=False):
        """
        Args:
            thresholds (`list[float]`):
                A list of thresholds used to stratify predictions into levels.
            labels (`list[int`):
                A list of values to label predictions belonging at each level. A label can be one of {-1, 0, 1}
                signifying {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (`bool`, *optional*, defaults to `False`):
                If `True`, produce additional matches for predictions with maximum match quality lower than
                high_threshold. See `set_low_quality_matches_` for more details.

            For example,
                thresholds = [0.3, 0.5] labels = [0, -1, 1] All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training. All predictions with 0.3 <= iou < 0.5 will
                be marked with -1 and thus will be ignored. All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        thresholds = thresholds[:]
        if thresholds[0] < 0:
            raise ValueError('Thresholds should be positive')
        thresholds.insert(0, -float('inf'))
        thresholds.append(float('inf'))
        if not all((low <= high for low, high in zip(thresholds[:-1], thresholds[1:]))):
            raise ValueError('Thresholds should be sorted.')
        if not all((l in [-1, 0, 1] for l in labels)):
            raise ValueError('All labels should be either -1, 0 or 1')
        if len(labels) != len(thresholds) - 1:
            raise ValueError('Number of labels should be equal to number of thresholds - 1')
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted elements. All elements must be >= 0
                (due to the us of `torch.nonzero` for selecting indices in `set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full((match_quality_matrix.size(1),), 0, dtype=torch.int64)
            default_match_labels = match_quality_matrix.new_full((match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8)
            return (default_matches, default_match_labels)
        assert torch.all(match_quality_matrix >= 0)
        matched_vals, matches = match_quality_matrix.max(dim=0)
        match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
        for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels[low_high] = l
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)
        return (matches, match_labels)

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches. Specifically, for each
        ground-truth G find the set of predictions that have maximum overlap with it (including ties); for each
        prediction in that set, if it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of :paper:`Faster R-CNN`.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        _, pred_inds_with_highest_quality = nonzero_tuple(match_quality_matrix == highest_quality_foreach_gt[:, None])
        match_labels[pred_inds_with_highest_quality] = 1