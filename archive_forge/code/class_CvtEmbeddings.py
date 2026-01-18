import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig
class CvtEmbeddings(nn.Module):
    """
    Construct the CvT embeddings.
    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding, dropout_rate):
        super().__init__()
        self.convolution_embeddings = CvtConvEmbeddings(patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, stride=stride, padding=padding)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, pixel_values):
        hidden_state = self.convolution_embeddings(pixel_values)
        hidden_state = self.dropout(hidden_state)
        return hidden_state