from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
def _extract_keypoints(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Based on their scores, extract the pixels that represent the keypoints that will be used for descriptors computation"""
    _, height, width = scores.shape
    keypoints = torch.nonzero(scores[0] > self.keypoint_threshold)
    scores = scores[0][tuple(keypoints.t())]
    keypoints, scores = remove_keypoints_from_borders(keypoints, scores, self.border_removal_distance, height * 8, width * 8)
    if self.max_keypoints >= 0:
        keypoints, scores = top_k_keypoints(keypoints, scores, self.max_keypoints)
    keypoints = torch.flip(keypoints, [1]).float()
    return (keypoints, scores)