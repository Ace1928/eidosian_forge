import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_trajectory_transformer import TrajectoryTransformerConfig
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.l1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.l2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]], layer_past: Optional[Tuple[torch.Tensor]]=None, use_cache: Optional[bool]=False, output_attentions: Optional[bool]=False):
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_outputs = self.attn(hidden_states, layer_past=layer_past, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.l1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.l2(hidden_states)
        hidden_states = residual + self.drop(hidden_states)
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs