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
class MgpstrA3Module(nn.Module):

    def __init__(self, config: MgpstrConfig):
        super().__init__()
        self.token_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.tokenLearner = nn.Sequential(nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False), nn.Conv2d(config.hidden_size, config.max_token_length, kernel_size=(1, 1), stride=1, bias=False))
        self.feat = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.token_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
        selected = self.tokenLearner(hidden_states)
        selected = selected.flatten(2)
        attentions = F.softmax(selected, dim=-1)
        feat = self.feat(hidden_states)
        feat = feat.flatten(2).transpose(1, 2)
        feat = torch.einsum('...si,...id->...sd', attentions, feat)
        a3_out = self.norm(feat)
        return (a3_out, attentions)