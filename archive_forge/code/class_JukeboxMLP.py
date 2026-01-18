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
class JukeboxMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        hidden_dim = int(config.mlp_multiplier * embed_dim)
        self.c_fc = JukeboxConv1D(embed_dim, hidden_dim)
        self.c_proj = JukeboxConv1D(hidden_dim, embed_dim)
        self.act = ACT2FN[config.act_fn]
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states