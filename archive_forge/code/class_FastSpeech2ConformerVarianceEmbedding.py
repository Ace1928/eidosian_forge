import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerVarianceEmbedding(nn.Module):

    def __init__(self, in_channels=1, out_channels=384, kernel_size=1, padding=0, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states