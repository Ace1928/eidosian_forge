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
class CvtSelfAttentionProjection(nn.Module):

    def __init__(self, embed_dim, kernel_size, padding, stride, projection_method='dw_bn'):
        super().__init__()
        if projection_method == 'dw_bn':
            self.convolution_projection = CvtSelfAttentionConvProjection(embed_dim, kernel_size, padding, stride)
        self.linear_projection = CvtSelfAttentionLinearProjection()

    def forward(self, hidden_state):
        hidden_state = self.convolution_projection(hidden_state)
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state