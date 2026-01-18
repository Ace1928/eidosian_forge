import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerPatchEmbeddings(nn.Module):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """

    def __init__(self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool=True):
        super().__init__()
        self.num_channels = num_channels
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=config.downsample_patch_size, stride=config.downsample_stride, padding=config.downsample_pad)
        self.norm = nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps) if apply_norm else nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embeddings = self.projection(pixel_values)
        embeddings = self.norm(embeddings)
        return embeddings