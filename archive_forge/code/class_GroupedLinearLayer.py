import math
import os
from operator import attrgetter
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, get_activation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_convbert import ConvBertConfig
class GroupedLinearLayer(nn.Module):

    def __init__(self, input_size, output_size, num_groups):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups
        self.weight = nn.Parameter(torch.empty(self.num_groups, self.group_in_dim, self.group_out_dim))
        self.bias = nn.Parameter(torch.empty(output_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = list(hidden_states.size())[0]
        x = torch.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim])
        x = x.permute(1, 0, 2)
        x = torch.matmul(x, self.weight)
        x = x.permute(1, 0, 2)
        x = torch.reshape(x, [batch_size, -1, self.output_size])
        x = x + self.bias
        return x