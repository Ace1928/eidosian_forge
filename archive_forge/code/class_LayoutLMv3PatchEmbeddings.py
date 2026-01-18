import collections
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
class LayoutLMv3PatchEmbeddings(nn.Module):
    """LayoutLMv3 image (patch) embeddings. This class also automatically interpolates the position embeddings for varying
    image sizes."""

    def __init__(self, config):
        super().__init__()
        image_size = config.input_size if isinstance(config.input_size, collections.abc.Iterable) else (config.input_size, config.input_size)
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values, position_embedding=None):
        embeddings = self.proj(pixel_values)
        if position_embedding is not None:
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1)
            position_embedding = position_embedding.permute(0, 3, 1, 2)
            patch_height, patch_width = (embeddings.shape[2], embeddings.shape[3])
            position_embedding = F.interpolate(position_embedding, size=(patch_height, patch_width), mode='bicubic')
            embeddings = embeddings + position_embedding
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings