import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
class FunnelClassificationHead(nn.Module):

    def __init__(self, config: FunnelConfig, n_labels: int) -> None:
        super().__init__()
        self.linear_hidden = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.linear_out = nn.Linear(config.d_model, n_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.linear_hidden(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.dropout(hidden)
        return self.linear_out(hidden)