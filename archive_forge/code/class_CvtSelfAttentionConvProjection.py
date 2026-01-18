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
class CvtSelfAttentionConvProjection(nn.Module):

    def __init__(self, embed_dim, kernel_size, padding, stride):
        super().__init__()
        self.convolution = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=embed_dim)
        self.normalization = nn.BatchNorm2d(embed_dim)

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        return hidden_state