from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
@staticmethod
def _sample_descriptors(keypoints, descriptors, scale: int=8) -> torch.Tensor:
    """Interpolate descriptors at keypoint locations"""
    batch_size, num_channels, height, width = descriptors.shape
    keypoints = keypoints - scale / 2 + 0.5
    divisor = torch.tensor([[width * scale - scale / 2 - 0.5, height * scale - scale / 2 - 0.5]])
    divisor = divisor.to(keypoints)
    keypoints /= divisor
    keypoints = keypoints * 2 - 1
    kwargs = {'align_corners': True} if is_torch_greater_or_equal_than_1_13 else {}
    keypoints = keypoints.view(batch_size, 1, -1, 2)
    descriptors = nn.functional.grid_sample(descriptors, keypoints, mode='bilinear', **kwargs)
    descriptors = descriptors.reshape(batch_size, num_channels, -1)
    descriptors = nn.functional.normalize(descriptors, p=2, dim=1)
    return descriptors