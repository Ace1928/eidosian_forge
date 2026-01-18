import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
class Swin2SRPatchEmbeddings(nn.Module):

    def __init__(self, config, normalize_patches=True):
        super().__init__()
        num_channels = config.embed_dim
        image_size, patch_size = (config.image_size, config.patch_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.projection = nn.Conv2d(num_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layernorm = nn.LayerNorm(config.embed_dim) if normalize_patches else None

    def forward(self, embeddings: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        embeddings = self.projection(embeddings)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings)
        return (embeddings, output_dimensions)