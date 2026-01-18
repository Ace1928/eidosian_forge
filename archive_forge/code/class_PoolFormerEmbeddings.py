import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_poolformer import PoolFormerConfig
class PoolFormerEmbeddings(nn.Module):
    """
    Construct Patch Embeddings.
    """

    def __init__(self, hidden_size, num_channels, patch_size, stride, padding, norm_layer=None):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(hidden_size) if norm_layer else nn.Identity()

    def forward(self, pixel_values):
        embeddings = self.projection(pixel_values)
        embeddings = self.norm(embeddings)
        return embeddings