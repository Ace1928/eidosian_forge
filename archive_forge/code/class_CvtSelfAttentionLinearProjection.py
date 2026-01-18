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
class CvtSelfAttentionLinearProjection(nn.Module):

    def forward(self, hidden_state):
        batch_size, num_channels, height, width = hidden_state.shape
        hidden_size = height * width
        hidden_state = hidden_state.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        return hidden_state