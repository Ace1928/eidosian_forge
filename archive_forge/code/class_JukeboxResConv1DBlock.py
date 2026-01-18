import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
class JukeboxResConv1DBlock(nn.Module):

    def __init__(self, config, conv_width, depth=1, res_scale=1.0):
        super().__init__()
        hidden_dim = config.res_convolution_multiplier * conv_width
        dilation = config.res_dilation_growth_rate ** depth
        padding = dilation
        self.res_scale = res_scale
        self.activation = nn.ReLU()
        self.conv1d_1 = nn.Conv1d(conv_width, hidden_dim, 3, 1, padding, dilation)
        self.conv1d_2 = nn.Conv1d(hidden_dim, conv_width, 1, 1, 0)

    def forward(self, hidden_states):
        residuals = hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv1d_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv1d_2(hidden_states)
        return residuals + self.res_scale * hidden_states