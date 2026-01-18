from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
class SuperPointInterestPointDecoder(nn.Module):
    """
    The SuperPointInterestPointDecoder uses the output of the SuperPointEncoder to compute the keypoint with scores.
    The scores are first computed by a convolutional layer, then a softmax is applied to get a probability distribution
    over the 65 possible keypoint classes. The keypoints are then extracted from the scores by thresholding and
    non-maximum suppression. Post-processing is then applied to remove keypoints too close to the image borders as well
    as to keep only the k keypoints with highest score.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        self.keypoint_threshold = config.keypoint_threshold
        self.max_keypoints = config.max_keypoints
        self.nms_radius = config.nms_radius
        self.border_removal_distance = config.border_removal_distance
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_score_a = nn.Conv2d(config.encoder_hidden_sizes[-1], config.decoder_hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_score_b = nn.Conv2d(config.decoder_hidden_size, config.keypoint_decoder_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self._get_pixel_scores(encoded)
        keypoints, scores = self._extract_keypoints(scores)
        return (keypoints, scores)

    def _get_pixel_scores(self, encoded: torch.Tensor) -> torch.Tensor:
        """Based on the encoder output, compute the scores for each pixel of the image"""
        scores = self.relu(self.conv_score_a(encoded))
        scores = self.conv_score_b(scores)
        scores = nn.functional.softmax(scores, 1)[:, :-1]
        batch_size, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch_size, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch_size, height * 8, width * 8)
        scores = simple_nms(scores, self.nms_radius)
        return scores

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