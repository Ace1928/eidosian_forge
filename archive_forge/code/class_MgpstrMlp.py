import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mgp_str import MgpstrConfig
class MgpstrMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, config: MgpstrConfig, hidden_features):
        super().__init__()
        hidden_features = hidden_features or config.hidden_size
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states