import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_segformer import SegformerConfig
class SegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return (embeddings, height, width)