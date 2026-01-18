from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ...utils import (
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_vipllava import VipLlavaConfig
class VipLlavaMultiModalProjector(nn.Module):

    def __init__(self, config: VipLlavaConfig):
        super().__init__()
        self.projector_layernorm = nn.LayerNorm(len(config.vision_feature_layers) * config.vision_config.hidden_size, eps=config.projector_layernorm_eps)
        self.linear_1 = nn.Linear(len(config.vision_feature_layers) * config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.projector_layernorm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states