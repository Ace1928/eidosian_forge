from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
class SuperPointDescriptorDecoder(nn.Module):
    """
    The SuperPointDescriptorDecoder uses the outputs of both the SuperPointEncoder and the
    SuperPointInterestPointDecoder to compute the descriptors at the keypoints locations.

    The descriptors are first computed by a convolutional layer, then normalized to have a norm of 1. The descriptors
    are then interpolated at the keypoints locations.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_descriptor_a = nn.Conv2d(config.encoder_hidden_sizes[-1], config.decoder_hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_descriptor_b = nn.Conv2d(config.decoder_hidden_size, config.descriptor_decoder_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, encoded: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Based on the encoder output and the keypoints, compute the descriptors for each keypoint"""
        descriptors = self.conv_descriptor_b(self.relu(self.conv_descriptor_a(encoded)))
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)
        descriptors = self._sample_descriptors(keypoints[None], descriptors[0][None], 8)[0]
        descriptors = torch.transpose(descriptors, 0, 1)
        return descriptors

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