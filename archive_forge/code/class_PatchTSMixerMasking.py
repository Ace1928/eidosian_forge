import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
class PatchTSMixerMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        config (`PatchTSMixerConfig`): model config
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.random_mask_ratio = config.random_mask_ratio
        self.channel_consistent_masking = config.channel_consistent_masking
        self.mask_type = config.mask_type
        self.num_forecast_mask_patches = config.num_forecast_mask_patches
        self.unmasked_channel_indices = config.unmasked_channel_indices
        self.mask_value = config.mask_value
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices = sorted(self.unmasked_channel_indices)

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input

        Return:
            masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points

        """
        if self.mask_type == 'random':
            masked_input, mask = random_masking(inputs=patch_input, mask_ratio=self.random_mask_ratio, unmasked_channel_indices=self.unmasked_channel_indices, channel_consistent_masking=self.channel_consistent_masking, mask_value=self.mask_value)
        elif self.mask_type == 'forecast':
            masked_input, mask = forecast_masking(inputs=patch_input, num_forecast_mask_patches=self.num_forecast_mask_patches, unmasked_channel_indices=self.unmasked_channel_indices, mask_value=self.mask_value)
        else:
            raise ValueError(f'Invalid mask type {self.mask_type}.')
        mask = mask.bool()
        return (masked_input, mask)