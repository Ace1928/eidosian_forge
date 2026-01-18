import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seggpt import SegGptConfig
from ..deprecated._archive_maps import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class SegGptLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.beta = config.beta
        self.patch_size = config.patch_size

    def forward(self, pixel_values: torch.FloatTensor, prompt_pixel_values: torch.FloatTensor, pred_masks: torch.FloatTensor, labels: torch.FloatTensor, bool_masked_pos: torch.BoolTensor):
        """Computes the L1 loss between the predicted masks and the ground truth masks.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, 2*height, width)`):
                Concatenated pixel values from prompt and input images.

            prompt_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, 2*height, width)`):
                Concatenated pixel values from mask prompt.

            pred_masks (`torch.FloatTensor` of shape `(batch_size, num_channels, 2*height, width)`):
                Predicted masks.

            labels (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Ground truth mask for input images.

            bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
                Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:
            `torch.FloatTensor`: The mean L1 loss between the predicted masks and the ground truth masks.
        """
        mask = bool_masked_pos[:, :, None].repeat(1, 1, self.patch_size ** 2 * 3)
        mask = unpatchify(mask, pixel_values.shape[1] // self.patch_size, pixel_values.shape[2] // self.patch_size)
        prompt_pixel_values = prompt_pixel_values.clone()
        prompt_pixel_values[:, :, prompt_pixel_values.shape[2] // 2:, :] = labels
        loss = F.smooth_l1_loss(pred_masks, prompt_pixel_values, reduction='none', beta=self.beta)
        loss = (loss * mask).sum() / mask.sum()
        return loss