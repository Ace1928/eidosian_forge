import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seggpt import SegGptConfig
from ..deprecated._archive_maps import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class SegGptLayer(nn.Module):

    def __init__(self, config: SegGptConfig, drop_path_rate: float) -> None:
        super().__init__()
        self.attention = SegGptAttention(config)
        self.mlp = SegGptMlp(config)
        self.drop_path = SegGptDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, ensemble_cond: int, feature_ensemble: bool=False, output_attentions: bool=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if feature_ensemble and attention_output.shape[0] // 2 >= ensemble_cond:
            prompt, inputs = attention_output.split(attention_output.shape[1] // 2, dim=1)
            if ensemble_cond == 2:
                num_prompts = attention_output.shape[0] // 2
                inputs = inputs.reshape(2, num_prompts, -1)
                inputs = inputs.mean(dim=1, keepdim=True).expand_as(inputs)
                inputs = inputs.reshape(*prompt.shape)
            else:
                inputs = inputs.mean(dim=0, keepdim=True).expand_as(inputs)
            attention_output = torch.cat([prompt, inputs], dim=1)
        hidden_states = self.drop_path(attention_output) + hidden_states
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.drop_path(hidden_states)
        outputs = (hidden_states,) + outputs
        return outputs