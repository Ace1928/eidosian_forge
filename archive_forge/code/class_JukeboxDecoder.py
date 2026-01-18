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
class JukeboxDecoder(nn.Module):

    def __init__(self, config, hidden_dim, depth, levels, downs_t, strides_t):
        super().__init__()
        self.levels = levels
        self.level_blocks = nn.ModuleList()
        for level, down_t, stride_t in zip(list(range(self.levels)), downs_t, strides_t):
            self.level_blocks.append(JukeboxDecoderConvBock(config, config.embed_dim, hidden_dim, depth, down_t, stride_t))
        self.out = nn.Conv1d(config.embed_dim, config.conv_input_shape, 3, 1, 1)

    def forward(self, hidden_states, all_levels=True):
        hidden_state = hidden_states[-1]
        for level in reversed(range(self.levels)):
            level_block = self.level_blocks[level]
            hidden_state = level_block(hidden_state)
            if level != 0 and all_levels:
                hidden_state = hidden_state + hidden_states[level - 1]
        hidden_state = self.out(hidden_state)
        return hidden_state