import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerPredictorLayer(nn.Module):

    def __init__(self, input_channels, num_chans, kernel_size, dropout_rate):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, num_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(num_chans)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = hidden_states.transpose(1, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, -1)
        hidden_states = self.dropout(hidden_states)
        return hidden_states