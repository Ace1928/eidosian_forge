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
class JukeboxBlock(nn.Module):

    def __init__(self, config, n_ctx, attn_func='dense_attn'):
        super().__init__()
        self.width = config.hidden_size
        self.attn = JukeboxAttention(config, n_ctx, attn_func=attn_func)
        self.layer_norm_0 = JukeboxLayerNorm(config.hidden_size)
        self.mlp = JukeboxMLP(config)
        self.layer_norm_1 = JukeboxLayerNorm(config.hidden_size)
        self.res_scale = 1.0 / config.num_layers if config.attn_res_scale else 1.0
        self.attn_func = attn_func

    def forward(self, hidden_states, last_encoder_hidden_states, sample=False):
        residuals = hidden_states
        hidden_states = self.layer_norm_0(hidden_states)
        hidden_states = self.attn(hidden_states, last_encoder_hidden_states, sample)
        output_states = self.layer_norm_1(residuals + hidden_states)
        output_states = self.mlp(output_states)
        if self.res_scale == 1.0:
            output = residuals + hidden_states + output_states
        else:
            output = residuals + self.res_scale * (hidden_states + output_states)
        return output