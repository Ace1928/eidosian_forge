import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_pvt_v2 import PvtV2Config
class PvtV2OverlapPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        patch_size = config.patch_sizes[layer_idx]
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        stride = config.strides[layer_idx]
        num_channels = config.num_channels if layer_idx == 0 else config.hidden_sizes[layer_idx - 1]
        hidden_size = config.hidden_sizes[layer_idx]
        self.patch_size = patch_size
        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return (embeddings, height, width)