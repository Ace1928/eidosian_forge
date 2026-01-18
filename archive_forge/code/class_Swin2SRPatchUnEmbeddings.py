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
class Swin2SRPatchUnEmbeddings(nn.Module):
    """Image to Patch Unembedding"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim

    def forward(self, embeddings, x_size):
        batch_size, height_width, num_channels = embeddings.shape
        embeddings = embeddings.transpose(1, 2).view(batch_size, self.embed_dim, x_size[0], x_size[1])
        return embeddings