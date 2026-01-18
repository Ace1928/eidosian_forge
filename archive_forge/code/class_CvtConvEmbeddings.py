import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig
class CvtConvEmbeddings(nn.Module):
    """
    Image to Conv Embedding.
    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        pixel_values = self.projection(pixel_values)
        batch_size, num_channels, height, width = pixel_values.shape
        hidden_size = height * width
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        if self.normalization:
            pixel_values = self.normalization(pixel_values)
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return pixel_values