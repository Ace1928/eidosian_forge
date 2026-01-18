import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_videomae import VideoMAEConfig
class VideoMAEPatchEmbeddings(nn.Module):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """

    def __init__(self, config):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv3d(in_channels=num_channels, out_channels=hidden_size, kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]), stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings