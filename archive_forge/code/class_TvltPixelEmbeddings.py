import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
class TvltPixelEmbeddings(nn.Module):
    """Construct the patch and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = TvltPixelPatchEmbeddings(config)
        self.num_patches_per_image = self.patch_embeddings.num_patches_per_image
        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, config.hidden_size))
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_per_image, config.hidden_size))
        self.config = config

    def forward(self, pixel_values, attention_masks=None):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        embeddings += self.pos_embed_v.repeat(1, num_frames, 1)
        embeddings += torch.repeat_interleave(self.temporal_embed[:, :num_frames], self.num_patches_per_image, dim=1)
        embeddings += self.type_embed_v
        return (embeddings, attention_masks)