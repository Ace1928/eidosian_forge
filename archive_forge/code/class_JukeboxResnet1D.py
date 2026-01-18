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
class JukeboxResnet1D(nn.Module):

    def __init__(self, config, conv_width, n_depth, reverse_dilation=False):
        super().__init__()
        self.dilation_cycle = config.res_dilation_cycle
        res_scale = 1.0 if not config.conv_res_scale else 1.0 / math.sqrt(n_depth)
        blocks = []
        for depth in range(n_depth):
            block_depth = depth if self.dilation_cycle is None else depth % self.dilation_cycle
            blocks.append(JukeboxResConv1DBlock(config, conv_width, block_depth, res_scale))
        if reverse_dilation:
            blocks = blocks[::-1]
        self.resnet_block = nn.ModuleList(blocks)

    def forward(self, hidden_states):
        for block in self.resnet_block:
            hidden_states = block(hidden_states)
        return hidden_states