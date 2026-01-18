import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils.backbone_utils import BackboneMixin
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = MaskFormerSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return (embeddings, output_dimensions)