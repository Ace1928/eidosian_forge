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
class JukeboxEncoderConvBlock(nn.Module):

    def __init__(self, config, embed_dim, hidden_dim, depth, down_t, stride_t):
        super().__init__()
        blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                blocks.append(nn.Conv1d(embed_dim if i == 0 else hidden_dim, hidden_dim, filter_t, stride_t, pad_t))
                blocks.append(JukeboxResnet1D(config, hidden_dim, depth))
        self.proj_out = nn.Conv1d(hidden_dim, config.embed_dim, 3, 1, 1)
        self.downsample_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        for block in self.downsample_block:
            hidden_states = block(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states